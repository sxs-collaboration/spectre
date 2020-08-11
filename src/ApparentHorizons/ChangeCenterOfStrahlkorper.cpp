// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/ChangeCenterOfStrahlkorper.hpp"

#include <cmath>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

// Get the r_hat vector of the Strahlkorper, which defines the
// collocation points.
template <typename Frame>
tnsr::i<DataVector, 3, Frame> get_rhat(
    const Strahlkorper<Frame>& strahlkorper) noexcept {
  const auto new_theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const DataVector sin_theta = sin(new_theta_phi[0]);
  return tnsr::i<DataVector, 3, Frame>{{sin_theta * cos(new_theta_phi[1]),
                                        sin_theta * sin(new_theta_phi[1]),
                                        cos(new_theta_phi[0])}};
}

// Work function called by the two interface functions.
template <typename Frame>
void change_expansion_center(
    const gsl::not_null<Strahlkorper<Frame>*> strahlkorper,
    const std::array<double, 3>& new_center,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept {
  // Get new_center minus old_center.
  const auto& old_center = strahlkorper->center();
  const std::array<double, 3> center_difference{
      {new_center[0] - old_center[0], new_center[1] - old_center[1],
       new_center[2] - old_center[2]}};

  // For bracketing the root, get the min and max radius with respect
  // to the new center.
  const auto [r_min, r_max] =
      [&center_difference, &r_hat, &strahlkorper ]() noexcept {
    const auto radius_old = strahlkorper->ylm_spherepack().spec_to_phys(
        strahlkorper->coefficients());
    DataVector radius_new =
        square(get<0>(r_hat) * radius_old - center_difference[0]);
    for (size_t d = 1; d < 3; ++d) {  // Start at 1; already did zero.
      radius_new +=
          square(r_hat.get(d) * radius_old - gsl::at(center_difference, d));
    }
    radius_new = sqrt(radius_new);
    const auto minmax =
        std::minmax_element(radius_new.begin(), radius_new.end());
    return std::make_pair(*(minmax.first), *(minmax.second));
  }
  ();

  // Find the coordinate radius of the surface, with respect to the
  // new center, at each of the angular collocation points of the
  // surface with respect to the new center. To do so, for each index
  // 's' (corresponding to an angular collocation point), find the
  // root 'r_new' that zeroes this lambda function.
  const auto radius_function = [&center_difference, &r_hat, &strahlkorper ](
      const double r_new, const size_t s) noexcept {
    // Get cartesian coordinates of the point with respect to the old
    // center.
    const std::array<double, 3> x_old{
        {r_new * get<0>(r_hat)[s] + center_difference[0],
         r_new * get<1>(r_hat)[s] + center_difference[1],
         r_new * get<2>(r_hat)[s] + center_difference[2]}};

    // Find (r, theta, phi) of the same point with respect to the old_center.
    const double r_old =
        sqrt(square(x_old[0]) + square(x_old[1]) + square(x_old[2]));
    const double theta_old = acos(x_old[2] / r_old);
    const double phi_old = atan2(x_old[1], x_old[0]);

    // Evaluate the radius of the surface on the old strahlkorper
    // at theta_old, phi_old.
    const double r_surf_old = strahlkorper->radius(theta_old, phi_old);

    // If r_surf_old is r_old, then 'r_new' is on the surface.
    return r_surf_old - r_old;
  };

  // We try to bracket the root between r_min and r_max.
  // But r_min and r_max are only approximate, and there may be grid points
  // with radii outside that range. So we pad by 10% to be safe.
  const double padding = 0.10;

  const auto radius_at_each_angle = RootFinder::toms748(
      radius_function,
      make_with_value<DataVector>(get<0>(r_hat), r_min * (1.0 - padding)),
      make_with_value<DataVector>(get<0>(r_hat), r_max * (1.0 + padding)),
      std::numeric_limits<double>::epsilon() * (r_min + r_max),
      2.0 * std::numeric_limits<double>::epsilon());

  // Now reset the radius and center of the new strahlkorper.
  *strahlkorper =
      Strahlkorper<Frame>(strahlkorper->l_max(), strahlkorper->m_max(),
                          radius_at_each_angle, new_center);
}
}  // namespace

template <typename Frame>
void change_expansion_center_of_strahlkorper(
    const gsl::not_null<Strahlkorper<Frame>*> strahlkorper,
    const std::array<double, 3>& new_center) noexcept {
  change_expansion_center(strahlkorper, new_center, get_rhat(*strahlkorper));
}

template <typename Frame>
void change_expansion_center_of_strahlkorper_to_physical(
    const gsl::not_null<Strahlkorper<Frame>*> strahlkorper) noexcept {
  const auto r_hat = get_rhat(*strahlkorper);

  // Zeroth iteration.
  change_expansion_center(strahlkorper, strahlkorper->physical_center(), r_hat);

  // In the random number tests, it never needed more than 7
  // iterations to converge to relative error of roundoff.  Allow 14
  // iterations to be safe.
  const size_t maxiter = 14;
  for (size_t iter = 0; iter < maxiter; ++iter) {
    const auto phys_center = strahlkorper->physical_center();
    const auto center = strahlkorper->center();
    const auto average_radius = strahlkorper->average_radius();
    const double relative_error = sqrt(square(center[0] - phys_center[0]) +
                                       square(center[1] - phys_center[1]) +
                                       square(center[2] - phys_center[2])) /
                                  average_radius;
    if (relative_error <= 2.0 * std::numeric_limits<double>::epsilon()) {
      return;
    }
    change_expansion_center(strahlkorper, phys_center, r_hat);
  }
  ERROR("Too many iterations in change_expansion_center_of_strahlkorper");
}
/// \cond
#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                         \
  template void change_expansion_center_of_strahlkorper(             \
      const gsl::not_null<Strahlkorper<FRAME(data)>*> strahlkorper,  \
      const std::array<double, 3>& new_center) noexcept;             \
  template void change_expansion_center_of_strahlkorper_to_physical( \
      const gsl::not_null<Strahlkorper<FRAME(data)>*> strahlkorper) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (::Frame::Inertial))

#undef INSTANTIATE
#undef FRAME
/// \endcond
