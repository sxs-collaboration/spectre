// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeCoords.hpp"

#include <cstddef>
#include <deque>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
template <typename Fr>
bool set_current_iteration_coords(
    const gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration,
    const LinkedMessageId<double>& time, const FastFlow& fast_flow,
    const ylm::Strahlkorper<Fr>& initial_guess,
    const ylm::Strahlkorper<Fr>& previous_iteration_surface,
    const std::deque<ah::Storage::PreviousSurface<Fr>>& previous_surfaces,
    const size_t max_compute_coords_retries, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  if (fast_flow.current_iteration() == 0) {
    // Need to set the first surface. If this is the very first, set it to the
    // initial guess. If not, use the previous horizon surface. The surface will
    // potentially be extrapolated below
    current_iteration->strahlkorper = UNLIKELY(previous_surfaces.empty())
                                          ? initial_guess
                                          : previous_surfaces.front().surface;

    // If we have zero previous_surfaces, then the initial guess is already
    // in strahlkorper, so do nothing.
    //
    // If we have one previous_surface, then we have had a successful
    // horizon find, and the initial guess for the next horizon find is already
    // in strahlkorper, so again we do nothing.
    //
    // If we have 2 valid previous_surfaces, then we set the initial guess
    // by linear extrapolation in time using the last 2 previous_surfaces.
    //
    // If we have 3 valid previous_surfaces, then we set the initial guess
    // by quadratic extrapolation in time using the last 3
    // previous_surfaces.
    //
    // For extrapolation, we assume that
    // * Expansion center of all the Strahlkorpers are equal.
    // * Maximum L of all the Strahlkorpers are equal. It is easy to relax the
    //   max L assumption once we start adaptively changing the L of the
    //   strahlkorpers.
    if (LIKELY(previous_surfaces.size() > 2)) {
      // Quadratic extrapolation
      const double dt_0 = previous_surfaces[0].time.id - time.id;
      const double dt_1 = previous_surfaces[1].time.id - time.id;
      const double dt_2 = previous_surfaces[2].time.id - time.id;
      const double fac_0 = dt_1 * dt_2 / ((dt_1 - dt_0) * (dt_2 - dt_0));
      const double fac_1 = dt_0 * dt_2 / ((dt_2 - dt_1) * (dt_0 - dt_1));
      const double fac_2 = 1.0 - fac_0 - fac_1;
      current_iteration->strahlkorper.coefficients() =
          fac_0 * previous_surfaces[0].surface.coefficients() +
          fac_1 * previous_surfaces[1].surface.coefficients() +
          fac_2 * previous_surfaces[2].surface.coefficients();
    } else if (previous_surfaces.size() > 1) {
      // Linear extrapolation
      const double dt_0 = previous_surfaces[0].time.id - time.id;
      const double dt_1 = previous_surfaces[1].time.id - time.id;
      const double fac_0 = dt_1 / (dt_1 - dt_0);
      const double fac_1 = 1.0 - fac_0;
      current_iteration->strahlkorper.coefficients() =
          fac_0 * previous_surfaces[0].surface.coefficients() +
          fac_1 * previous_surfaces[1].surface.coefficients();
    }
  }

  const auto set_coords = [&]() {
    const auto& strahlkorper = current_iteration->strahlkorper;

    const size_t l_mesh = fast_flow.current_l_mesh(strahlkorper);

    const auto prolonged_strahlkorper =
        ylm::Strahlkorper<Fr>(l_mesh, l_mesh, strahlkorper);

    // Frames are handled within block_logical_coordinates
    current_iteration->block_coord_holders = ::block_logical_coordinates(
        domain, ylm::cartesian_coords(prolonged_strahlkorper), time.id,
        functions_of_time);
  };

  // Set the coordinates
  set_coords();

  // Check if any points were outside the domain
  current_iteration->compute_coords_retries = 0;
  while (alg::any_of(
      current_iteration->block_coord_holders.value(),
      [](const auto& coord_holder) { return not coord_holder.has_value(); })) {
    // If so, try recomputation up to max_compute_coords_retries times
    ++current_iteration->compute_coords_retries;

    if (current_iteration->compute_coords_retries <=
        max_compute_coords_retries) {
      if (fast_flow.current_iteration() == 0) {
        // If this is the zeroth iteration and we couldn't compute the coords,
        // then just try increasing the size of the horizon by 50%
        current_iteration->strahlkorper.coefficients()[0] *= 1.5;
      } else {
        // Otherwise move the new trial surface halfway between the current
        // surface and the previous surface
        current_iteration->strahlkorper.coefficients() +=
            0.5 * (previous_iteration_surface.coefficients() -
                   current_iteration->strahlkorper.coefficients());
      }

      // Set the coords using the new guess
      set_coords();
    } else {
      // We didn't actually try computation here so we added one extra to the
      // counter which we need to remove
      --current_iteration->compute_coords_retries;
      return false;
    }
  }

  return true;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template bool set_current_iteration_coords(                           \
      const gsl::not_null<ah::Storage::Iteration<FRAME(data)>*>         \
          current_iteration,                                            \
      const LinkedMessageId<double>& time, const FastFlow& fast_flow,   \
      const ylm::Strahlkorper<FRAME(data)>& initial_guess,              \
      const ylm::Strahlkorper<FRAME(data)>& previous_iteration_surface, \
      const std::deque<ah::Storage::PreviousSurface<FRAME(data)>>&      \
          previous_surfaces,                                            \
      const size_t max_compute_coords_retries, const Domain<3>& domain, \
      const domain::FunctionsOfTimeMap& functions_of_time);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah
