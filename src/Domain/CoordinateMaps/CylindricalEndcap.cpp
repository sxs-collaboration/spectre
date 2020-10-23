// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CylindricalEndcap.hpp"

#include <optional>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedEndcap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::CoordinateMaps {

CylindricalEndcap::CylindricalEndcap(const std::array<double, 3>& center_one,
                                     const std::array<double, 3>& center_two,
                                     const std::array<double, 3>& proj_center,
                                     double radius_one, double radius_two,
                                     double z_plane) noexcept
    : impl_(center_two, proj_center, radius_two,
            FocallyLiftedInnerMaps::Endcap(center_one, radius_one, z_plane)) {
#ifdef SPECTRE_DEBUG
  // There are two types of sanity checks here on the map parameters.
  // 1) ASSERTS that guarantee that the map is invertible.
  // 2) ASSERTS that guarantee that the map parameters fall within
  //    the range tested by the unit tests (which is the range in which
  //    the map is expected to be used).
  //
  // There are two reasons why 1) and 2) are not the same:
  //
  // a) It is possible to choose parameters such that the map is
  //    invertible but the resulting geometry has very sharp angles,
  //    very large or ill-conditioned Jacobians, or both.  We want to
  //    avoid such cases.
  // b) We do not want to waste effort testing the map for parameters
  //    that we don't expect to be used.  For example, we demand
  //    here that proj_center and sphere_one are contained within
  //    sphere_two, but the map should still be valid for some choices
  //    of parameters where sphere_one and sphere_two are disjoint;
  //    allowing those parameter choices would involve much more
  //    complicated logic to determine whether the map produces shapes
  //    with sharp angles or large jacobians, and it would involve more
  //    complicated unit tests to cover those possibilities.

  // First test for invertibility.

  // Consider the intersection of sphere_one and the plane formed by
  // z_plane.  Call it circle_one.  Consider the cone with apex
  // center_one that intersects sphere_one on circle_one. This cone
  // has an opening angle 2*theta.  Call the cone 'cone_one'.
  const double cos_theta = (z_plane - center_one[2]) / radius_one;

  // Now consider a different cone, called 'invertibility_cone', constructed so
  // that invertibility_cone and cone_one intersect each other on circle_one at
  // right angles.  The apex of invertibility_cone is invertibility_cone_apex,
  // defined as follows:
  const std::array<double, 3> invertibility_cone_apex = {
      center_one[0], center_one[1], center_one[2] + radius_one / cos_theta};
  // invertibility_cone opens in the -z direction with opening angle
  // 2*(pi/2-theta). invertibility_cone_apex is labeled 'S' in the 'Allowed
  // Region' figure in CylindricalEndcap.hpp

  // A necessary condition for invertibility is that proj_center lies
  // either inside of invertibility_cone or inside of the reflection of
  // invertibility_cone (a cone with apex invertibility_cone_apex but opening in
  // the +x direction with opening angle 2*(pi/2-theta)).

  // Determine the angle of proj_center relative to invertibility_cone_apex.
  // Call this angle alpha.
  const double dist_cone_proj =
      sqrt(square(invertibility_cone_apex[0] - proj_center[0]) +
           square(invertibility_cone_apex[1] - proj_center[1]) +
           square(invertibility_cone_apex[2] - proj_center[2]));
  const double cos_alpha =
      (invertibility_cone_apex[2] - proj_center[2]) / dist_cone_proj;

  // Now make sure that alpha < pi/2-theta.
  // The cone on either side of invertibility_cone_apex is ok, so we use abs
  // below.
  ASSERT(acos(abs(cos_alpha)) < abs(asin(cos_theta)),
         "The arguments passed into the CylindricalEndcap constructor "
         "yield a noninvertible map.");

  // Another necessary condition for invertibility is that proj_center
  // cannot lie between sphere_one and invertibility_cone_apex.
  const double proj_radius_one_squared =
      square(center_one[0] - proj_center[0]) +
      square(center_one[1] - proj_center[1]) +
      square(center_one[2] - proj_center[2]);
  ASSERT(proj_center[2] > invertibility_cone_apex[2] or
             proj_center[2] < z_plane or
             proj_radius_one_squared < square(radius_one),
         "The arguments passed into the CylindricalEndcap constructor "
         "yield a noninvertible map.");

  // Other sanity checks that may be relaxed if there is more logic
  // added and more unit tests to test these cases.
  // We put these sanity checks in a separate place precisely because
  // they may be relaxed in the future, whereas the above sanity checks
  // are hard limits that shouldn't be touched.

  ASSERT(proj_center[2] < z_plane,
         "The map hasn't been tested for the "
         "case where proj_center is at larger z than the z_plane. "
         "The map may still be invertible, but further "
         "testing would be needed to ensure that jacobians are not "
         "ill-conditioned.");

  ASSERT(abs(cos_theta) <= 0.9,
         "z_plane is too far from the center of sphere_one. "
             << "cos_theta = " << cos_theta
             << ". If |cos_theta| > 1 the map is singular.  If 0.9 < "
                "|cos_theta| < 1 then the map is not singular, but the "
                "jacobians are likely to be large and the map has not been "
                "tested for these parameters.");
  ASSERT(abs(cos_theta) >= 0.1,
         "z_plane is too close to the center of sphere_one. "
             << "cos_theta = " << cos_theta
             << ". The map is not singular, but the jacobians are likely to be "
                "large and the map has not been tested for this choice of "
                "parameters.");

  const double dist_spheres = sqrt(square(center_one[0] - center_two[0]) +
                                   square(center_one[1] - center_two[1]) +
                                   square(center_one[2] - center_two[2]));
  ASSERT(dist_spheres + radius_one < radius_two,
         "The map has been tested only for the case when "
         "sphere_one is contained inside sphere_two");

  const double proj_radius_two_squared =
      square(center_two[0] - proj_center[0]) +
      square(center_two[1] - proj_center[1]) +
      square(center_two[2] - proj_center[2]);
  ASSERT(proj_radius_two_squared < square(radius_two),
         "The map has been tested only for the case when "
         "proj_center is contained inside sphere_two");

  // Check if we are too close to singular.  We do this check
  // separately from the earlier similar check because this check may
  // be changed in the future based on further testing and further
  // logic that may be added (for example we may want to change the
  // 0.95 to some other number), but the previous check (without the
  // 0.95) is a hard limit that should always be obeyed.
  ASSERT(acos(abs(cos_alpha)) < abs(0.95 * asin(cos_theta)),
         "Parameters are close to where the map becomes "
         "non-invertible.  The map has not been tested for this case.");

  // Check if opening angle is small enough.
  const double max_opening_angle = M_PI / 3.0;
  const double max_proj_center_z = z_plane - radius_one *
                                                 sqrt(1.0 - square(cos_theta)) /
                                                 tan(max_opening_angle);
  ASSERT(proj_center[2] < max_proj_center_z,
         "proj_center is too close to z_plane. The "
         "map has not been tested for this case.");
  // Let a line segment be drawn between the projection point and any
  // point on the intersection circle (the circle where sphere 1
  // intersects z_plane). Consider the angle between that line segment
  // and the z axis, and let beta be the maximum of that angle (over all points
  // on the intersection circle).  We demand that beta is less than a certain
  // maximum opening angle.
  const double tan_beta = sqrt(square(center_one[0] - proj_center[0]) +
                               square(center_one[1] - proj_center[1])) /
                          (max_proj_center_z - proj_center[2]);
  ASSERT(tan_beta < tan(max_opening_angle),
         "Opening angle is too large. The map has not "
         "been tested for this case.");
#endif
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> CylindricalEndcap::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  return impl_.operator()(source_coords);
}

std::optional<std::array<double, 3>> CylindricalEndcap::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  return impl_.inverse(target_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalEndcap::jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return impl_.jacobian(source_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalEndcap::inv_jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return impl_.inv_jacobian(source_coords);
}

void CylindricalEndcap::pup(PUP::er& p) noexcept { p | impl_; }

bool operator==(const CylindricalEndcap& lhs,
                const CylindricalEndcap& rhs) noexcept {
  return lhs.impl_ == rhs.impl_;
}
bool operator!=(const CylindricalEndcap& lhs,
                const CylindricalEndcap& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                 \
  CylindricalEndcap::operator()(                                               \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  CylindricalEndcap::jacobian(const std::array<DTYPE(data), 3>& source_coords) \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  CylindricalEndcap::inv_jacobian(                                             \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond

}  // namespace domain::CoordinateMaps
