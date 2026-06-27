// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeCoords.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
template <typename Fr>
bool set_current_iteration_coords(
    const gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration,
    const gsl::not_null<std::vector<size_t>*> block_order,
    const LinkedMessageId<double>& time, const FastFlow& fast_flow,
    const ylm::Strahlkorper<Fr>& initial_guess,
    const ylm::Strahlkorper<Fr>& previous_iteration_surface,
    const std::deque<ah::Storage::PreviousSurface<Fr>>& previous_surfaces,
    const size_t max_compute_coords_retries, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const std::optional<size_t>& current_resolution_l,
    const bool rerunning_with_higher_resolution) {
  // If we are rerunning the horizon find with higher resolution, use
  // the previous horizon as the initial guess.
  if (fast_flow.current_iteration() == 0) {
    if (rerunning_with_higher_resolution) {
      if (not current_resolution_l.has_value()) {
        ERROR(
            "Current resolution L is not set, even though this iteration is "
            "marked as a rerun with higher resolution. This is an internal "
            "error. Please file an issue.");
      }
      if (previous_iteration_surface.l_max() >= *current_resolution_l) {
        ERROR("Previous iteration surface has resolution "
              << previous_iteration_surface.l_max()
              << " but current resolution " << *current_resolution_l
              << " is not higher, even though this iteration is marked as a "
                 "rerun with higher resolution. This is an internal error. "
                 "Please file an issue.");
      }
      // Just set the initial guess by prolonging or restricting the previous
      // iteration surface, i.e., the horizon already found at a lower
      // resolution
      current_iteration->strahlkorper =
          ylm::Strahlkorper<Fr>(*current_resolution_l, *current_resolution_l,
                                previous_iteration_surface);
    } else {
      // Need to set the first surface. If this is the very first, set it to the
      // initial guess. If not, use the previous horizon surface. The surface
      // will potentially be extrapolated below.
      //
      // If current_resolution_l is set, make sure the initial guess has the
      // correct resolution. Only prolong or restrict if current_resolution_l
      // is different from the initial guess / previous surface l_max().
      // Otherwise, just set the initial guess.
      if (current_resolution_l.has_value()) {
        if (UNLIKELY(previous_surfaces.empty())) {
          if (UNLIKELY(current_resolution_l != initial_guess.l_max())) {
            current_iteration->strahlkorper = ylm::Strahlkorper<Fr>(
                *current_resolution_l, *current_resolution_l, initial_guess);
          } else {
            current_iteration->strahlkorper = initial_guess;
          }
        } else {
          if (UNLIKELY(current_resolution_l !=
                       previous_surfaces.front().surface.l_max())) {
            current_iteration->strahlkorper = ylm::Strahlkorper<Fr>(
                *current_resolution_l, *current_resolution_l,
                previous_surfaces.front().surface);
          } else {
            current_iteration->strahlkorper = previous_surfaces.front().surface;
          }
        }
      } else {
        current_iteration->strahlkorper =
            UNLIKELY(previous_surfaces.empty())
                ? initial_guess
                : previous_surfaces.front().surface;
      }

      // If we have zero previous_surfaces, then the initial guess is already
      // in strahlkorper, so do nothing.
      //
      // If we have one previous_surface, then we have had a successful
      // horizon find, and the initial guess for the next horizon find is
      // already in strahlkorper, so again we do nothing.
      //
      // If we have 2 valid previous_surfaces, then we set the initial guess
      // by linear extrapolation in time using the last 2 previous_surfaces.
      //
      // If we have 3 valid previous_surfaces, then we set the initial guess
      // by quadratic extrapolation in time using the last 3
      // previous_surfaces.
      //
      // For extrapolation, we assume that the expansion center of all the
      // Strahlkorpers are equal. If any surface has a different resolution,
      // prolong to the max resolution, extrapolate, then restrict to either
      // current_resolution.
      if (LIKELY(previous_surfaces.size() > 2)) {
        constexpr size_t number_of_previous_surfaces = 3;
        // Quadratic extrapolation
        const double dt_0 = previous_surfaces[0].time.id - time.id;
        const double dt_1 = previous_surfaces[1].time.id - time.id;
        const double dt_2 = previous_surfaces[2].time.id - time.id;
        const double fac_0 = dt_1 * dt_2 / ((dt_1 - dt_0) * (dt_2 - dt_0));
        const double fac_1 = dt_0 * dt_2 / ((dt_2 - dt_1) * (dt_0 - dt_1));
        const double fac_2 = 1.0 - fac_0 - fac_1;

        // Determine if resolutions are equal and, if not, which is greatest
        const auto [min_res_it, max_res_it] = std::minmax_element(
            previous_surfaces.begin(),
            previous_surfaces.begin() + number_of_previous_surfaces,
            [](const auto& a, const auto& b) {
              return a.surface.l_max() < b.surface.l_max();
            });
        const size_t min_resolution_l = min_res_it->surface.l_max();
        const size_t max_resolution_l = max_res_it->surface.l_max();
        const auto which_surface_is_max = static_cast<size_t>(
            std::distance(previous_surfaces.begin(), max_res_it));

        if (UNLIKELY(min_resolution_l != max_resolution_l)) {
          const std::array facs{fac_0, fac_1, fac_2};
          DataVector new_coefs =
              gsl::at(facs, which_surface_is_max) *
              gsl::at(previous_surfaces, which_surface_is_max)
                  .surface.coefficients();
          for (size_t i = 0; i < number_of_previous_surfaces; ++i) {
            if (i == which_surface_is_max) {
              continue;
            }
            new_coefs +=
                gsl::at(facs, i) *
                gsl::at(previous_surfaces, i)
                    .surface.ylm_spherepack()
                    .prolong_or_restrict(
                        gsl::at(previous_surfaces, i).surface.coefficients(),
                        gsl::at(previous_surfaces, which_surface_is_max)
                            .surface.ylm_spherepack());
          }
          current_iteration->strahlkorper.coefficients() =
              gsl::at(previous_surfaces, which_surface_is_max)
                  .surface.ylm_spherepack()
                  .prolong_or_restrict(
                      new_coefs,
                      current_iteration->strahlkorper.ylm_spherepack());
        } else {
          if (current_iteration->strahlkorper.l_max() !=
              previous_surfaces[0].surface.l_max()) {
            current_iteration->strahlkorper.coefficients() =
                previous_surfaces[0]
                    .surface.ylm_spherepack()
                    .prolong_or_restrict(
                        fac_0 * previous_surfaces[0].surface.coefficients() +
                            fac_1 *
                                previous_surfaces[1].surface.coefficients() +
                            fac_2 * previous_surfaces[2].surface.coefficients(),
                        current_iteration->strahlkorper.ylm_spherepack());
          } else {
            current_iteration->strahlkorper.coefficients() =
                fac_0 * previous_surfaces[0].surface.coefficients() +
                fac_1 * previous_surfaces[1].surface.coefficients() +
                fac_2 * previous_surfaces[2].surface.coefficients();
          }
        }
      } else if (previous_surfaces.size() > 1) {
        // Linear extrapolation
        const double dt_0 = previous_surfaces[0].time.id - time.id;
        const double dt_1 = previous_surfaces[1].time.id - time.id;
        const double fac_0 = dt_1 / (dt_1 - dt_0);
        const double fac_1 = 1.0 - fac_0;

        const size_t l_0 = previous_surfaces[0].surface.l_max();
        const size_t l_1 = previous_surfaces[1].surface.l_max();

        if (l_0 == l_1) {
          if (l_0 == current_iteration->strahlkorper.l_max()) {
            current_iteration->strahlkorper.coefficients() =
                fac_0 * previous_surfaces[0].surface.coefficients() +
                fac_1 * previous_surfaces[1].surface.coefficients();
          } else {
            current_iteration->strahlkorper.coefficients() =
                previous_surfaces[0]
                    .surface.ylm_spherepack()
                    .prolong_or_restrict(
                        fac_0 * previous_surfaces[0].surface.coefficients() +
                            fac_1 * previous_surfaces[1].surface.coefficients(),
                        current_iteration->strahlkorper.ylm_spherepack());
          }
        } else {
          const size_t which_surface_is_max = l_0 > l_1 ? 0 : 1;
          const size_t which_surface_is_min = 1 - which_surface_is_max;
          const std::array facs{fac_0, fac_1};
          DataVector new_coefs =
              gsl::at(facs, which_surface_is_max) *
              gsl::at(previous_surfaces, which_surface_is_max)
                  .surface.coefficients();
          new_coefs += gsl::at(facs, which_surface_is_min) *
                       gsl::at(previous_surfaces, which_surface_is_min)
                           .surface.ylm_spherepack()
                           .prolong_or_restrict(
                               gsl::at(previous_surfaces, which_surface_is_min)
                                   .surface.coefficients(),
                               gsl::at(previous_surfaces, which_surface_is_max)
                                   .surface.ylm_spherepack());
          current_iteration->strahlkorper.coefficients() =
              gsl::at(previous_surfaces, which_surface_is_max)
                  .surface.ylm_spherepack()
                  .prolong_or_restrict(
                      new_coefs,
                      current_iteration->strahlkorper.ylm_spherepack());
        }
      }
    }
  }  // if fastflow.current_iteration() == 0

  const auto set_coords = [&]() {
    const auto& strahlkorper = current_iteration->strahlkorper;

    const size_t l_mesh = fast_flow.current_l_mesh(strahlkorper);

    const auto prolonged_strahlkorper =
        ylm::Strahlkorper<Fr>(l_mesh, l_mesh, strahlkorper);

    // Frames are handled within block_logical_coordinates
    current_iteration->block_coord_holders = ::block_logical_coordinates(
        domain, ylm::cartesian_coords(prolonged_strahlkorper), time.id,
        functions_of_time, block_order);
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
      const gsl::not_null<std::vector<size_t>*> block_order,            \
      const LinkedMessageId<double>& time, const FastFlow& fast_flow,   \
      const ylm::Strahlkorper<FRAME(data)>& initial_guess,              \
      const ylm::Strahlkorper<FRAME(data)>& previous_iteration_surface, \
      const std::deque<ah::Storage::PreviousSurface<FRAME(data)>>&      \
          previous_surfaces,                                            \
      const size_t max_compute_coords_retries, const Domain<3>& domain, \
      const domain::FunctionsOfTimeMap& functions_of_time,              \
      const std::optional<size_t>& current_resolution_l,                \
      const bool rerunning_with_higher_resolution);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah
