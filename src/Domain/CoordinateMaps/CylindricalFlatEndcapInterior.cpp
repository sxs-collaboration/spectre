// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CylindricalFlatEndcapInterior.hpp"

#include <cmath>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedFlatEndcap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp11.hpp"

namespace domain::CoordinateMaps {

CylindricalFlatEndcapInterior::CylindricalFlatEndcapInterior(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two,
    const std::array<double, 3>& proj_center, const double z_sphere_extent,
    const double radius_two) {
  // Compute the flat disk radius from the desired sphere-cap extent.
  //
  // The focal projection from P through a disk rim point at radius r_disk
  // and z = center_one[2] hits the sphere at z = z_sphere_extent.
  // Parametrically the ray is: z(t) = proj_center[2] + t*(center_one[2] -
  // proj_center[2]), with t=1 at the disk.  The sphere is reached at
  //   t_sphere = (z_sphere_extent - proj_center[2])
  //              / (center_one[2]  - proj_center[2]),
  // and the transverse radius on the sphere is
  //   r_rim = sqrt(radius_two^2 - (z_sphere_extent - center_two[2])^2).
  // Therefore the disk radius is radius_one = r_rim / t_sphere.
  const double t_sphere =
      (z_sphere_extent - proj_center[2]) / (center_one[2] - proj_center[2]);
  const double dz_extent = z_sphere_extent - center_two[2];
  const double r_rim =
      std::sqrt(radius_two * radius_two - dz_extent * dz_extent);
  const double radius_one = r_rim / t_sphere;

  // source_is_between_focus_and_target = true: the flat disk lies between
  // the projection point P and the far wall of the sphere.
  impl_ = FocallyLiftedMap<FocallyLiftedInnerMaps::FlatEndcap>(
      center_two, proj_center, radius_two, true,
      FocallyLiftedInnerMaps::FlatEndcap(center_one, radius_one));

#ifdef SPECTRE_DEBUG
  // There are two types of sanity checks here on the map parameters.
  // 1) ASSERTS that guarantee that the map is invertible.
  // 2) ASSERTS that guarantee that the map parameters fall within
  //    the range tested by the unit tests (which is the range in which
  //    the map is expected to be used).

  ASSERT(
      std::abs(dz_extent) < radius_two,
      "z_sphere_extent ("
          << z_sphere_extent
          << ") must lie strictly on the sphere: "
             "|z_sphere_extent - center_two[2]| must be less than radius_two ("
          << radius_two << ").");

  ASSERT(
      t_sphere > 1.0,
      "The focal parameter t_sphere ("
          << t_sphere
          << ") must be greater than 1, meaning the sphere is on the far side "
             "of the disk from the projection point P.  Check that "
             "z_sphere_extent ("
          << z_sphere_extent << ") is beyond the disk at center_one[2] ("
          << center_one[2] << ") from the projection point at proj_center[2] ("
          << proj_center[2] << ").");

  const double dist_proj = std::sqrt(square(center_two[0] - proj_center[0]) +
                                     square(center_two[1] - proj_center[1]) +
                                     square(center_two[2] - proj_center[2]));
  ASSERT(dist_proj <= 0.95 * radius_two,
         "The map has been tested only for the case when proj_center is "
         "sufficiently inside the sphere (no closer than 95% of the way to "
         "the surface).");

  // The flat disk must lie inside the sphere, between 5% and 95% of
  // radius_two below the sphere center (in z).  This is the interior analogue
  // of CylindricalFlatEndcap's requirement that the flat disk lie outside the
  // sphere.
  ASSERT(center_one[2] >= center_two[2] - 0.95 * radius_two and
             center_one[2] <= center_two[2] - 0.05 * radius_two,
         "The map has only been tested for the case when the flat disk lies "
         "inside the sphere, at least 5% of radius_two below the sphere "
         "center and at most 95% of radius_two below it.");

  ASSERT(center_one[2] < proj_center[2],
         "The flat disk must be below the projection point (in z), so that "
         "the flat disk lies between P and the far wall of the sphere.");

  ASSERT(radius_one / radius_two <= 10.0 and radius_two / radius_one <= 10.0,
         "The map has been tested only for the case when the ratio of "
         "radius_one to radius_two is between 10 and 1/10.");

  // Every point on the flat disk rim must lie strictly inside the sphere.
  // The farthest rim point is at distance
  //   sqrt((|xy_offset| + R1)^2 + dz_disk^2) from center_two,
  // where xy_offset is the 2D separation of the disk and sphere centres and
  // dz_disk = center_one[2] - center_two[2].  If this exceeds R2 then some
  // rays from P through the disk hit the sphere at t < 1 (before the disk),
  // violating source_is_between_focus_and_target = true and making the
  // inverse undefined for those source points.
  const double xy_offset = std::sqrt(square(center_one[0] - center_two[0]) +
                                     square(center_one[1] - center_two[1]));
  const double dz_disk = center_one[2] - center_two[2];
  const double max_rim_dist =
      std::sqrt(square(xy_offset + radius_one) + square(dz_disk));
  ASSERT(max_rim_dist < radius_two,
         "The entire flat disk rim must lie strictly inside the sphere.  "
         "The farthest rim point is at distance "
             << max_rim_dist << " from center_two, but radius_two is "
             << radius_two
             << ".  Reduce the x,y offset between center_one and center_two, "
                "or reduce radius_one.");
#endif
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3>
CylindricalFlatEndcapInterior::operator()(
    const std::array<T, 3>& source_coords) const {
  // Negate zbar so that zbar=+1 maps to the flat disk and zbar=-1 maps to
  // the far sphere wall, keeping the Jacobian determinant positive.
  return impl_.operator()(std::array<tt::remove_cvref_wrap_t<T>, 3>{
      dereference_wrapper(source_coords[0]),
      dereference_wrapper(source_coords[1]),
      -dereference_wrapper(source_coords[2])});
}

std::optional<std::array<double, 3>> CylindricalFlatEndcapInterior::inverse(
    const std::array<double, 3>& target_coords) const {
  auto result = impl_.inverse(target_coords);
  if (result) {
    (*result)[2] = -(*result)[2];
  }
  return result;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalFlatEndcapInterior::jacobian(
    const std::array<T, 3>& source_coords) const {
  auto jac = impl_.jacobian(std::array<tt::remove_cvref_wrap_t<T>, 3>{
      dereference_wrapper(source_coords[0]),
      dereference_wrapper(source_coords[1]),
      -dereference_wrapper(source_coords[2])});
  // Chain rule: d/dzbar = -d/dzbar_eff, so negate column 2.
  for (size_t i = 0; i < 3; ++i) {
    jac.get(i, 2) = -jac.get(i, 2);
  }
  return jac;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalFlatEndcapInterior::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  auto inv_jac = impl_.inv_jacobian(std::array<tt::remove_cvref_wrap_t<T>, 3>{
      dereference_wrapper(source_coords[0]),
      dereference_wrapper(source_coords[1]),
      -dereference_wrapper(source_coords[2])});
  // Chain rule: dzbar/dx^i = -dzbar_eff/dx^i, so negate row 2.
  for (size_t i = 0; i < 3; ++i) {
    inv_jac.get(2, i) = -inv_jac.get(2, i);
  }
  return inv_jac;
}

void CylindricalFlatEndcapInterior::pup(PUP::er& p) { p | impl_; }

bool operator==(const CylindricalFlatEndcapInterior& lhs,
                const CylindricalFlatEndcapInterior& rhs) {
  return lhs.impl_ == rhs.impl_;
}

bool operator!=(const CylindricalFlatEndcapInterior& lhs,
                const CylindricalFlatEndcapInterior& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  CylindricalFlatEndcapInterior::operator()(                                 \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  CylindricalFlatEndcapInterior::jacobian(                                   \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  CylindricalFlatEndcapInterior::inv_jacobian(                               \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps
