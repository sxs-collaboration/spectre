// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CylindricalSide.hpp"

#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedSide.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::CoordinateMaps {

CylindricalSide::CylindricalSide(const std::array<double, 3>& center_one,
                                 const std::array<double, 3>& center_two,
                                 const std::array<double, 3>& proj_center,
                                 const double radius_one,
                                 const double radius_two, const double z_lower,
                                 const double z_upper) noexcept
    : impl_(
          center_two, proj_center, radius_two,
          [&center_one, &center_two, &radius_one, &radius_two]() noexcept {
            const double dist_spheres =
                sqrt(square(center_one[0] - center_two[0]) +
                     square(center_one[1] - center_two[1]) +
                     square(center_one[2] - center_two[2]));
            // If sphere 1 is contained in sphere 2, then the source (sphere 1)
            // is always between the projection point and the target (sphere 2).
            // Otherwise, if sphere 2 is contained in sphere 1, then the source
            // (sphere 1) is not between the projection point and the target
            // (sphere 2).
            //
            // Note that below we ASSERT that sphere 1 is contained in sphere 2
            // or vice versa.
            return dist_spheres + radius_one < radius_two;
          }(),
          FocallyLiftedInnerMaps::Side(center_one, radius_one, z_lower,
                                       z_upper)) {
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

  const double dist_spheres = sqrt(square(center_one[0] - center_two[0]) +
                                   square(center_one[1] - center_two[1]) +
                                   square(center_one[2] - center_two[2]));
  ASSERT(dist_spheres + radius_one < radius_two or
             dist_spheres + radius_two < radius_one,
         "CylindricalSide: The map has been tested only for the case when "
         "sphere_one is contained inside sphere_two or vice versa.");

  const double dist_proj_one = sqrt(square(center_one[0] - proj_center[0]) +
                                    square(center_one[1] - proj_center[1]) +
                                    square(center_one[2] - proj_center[2]));
  ASSERT(dist_proj_one < radius_one,
         "CylindricalSide: The map has been tested only for the case when "
         "proj_center is contained inside sphere_one");

  const double dist_proj_two = sqrt(square(center_two[0] - proj_center[0]) +
                                    square(center_two[1] - proj_center[1]) +
                                    square(center_two[2] - proj_center[2]));
  ASSERT(dist_proj_two < radius_two,
         "CylindricalSide: The map has been tested only for the case when "
         "proj_center is contained inside sphere_two. dist_proj_two="
         << dist_proj_two << ", radius_two=" << radius_two);

  ASSERT(proj_center[2] >= z_lower and proj_center[2] <= z_upper,
         "CylindricalSide: The map has been tested only for the case when "
         "proj_center is contained inside z_lower and z_upper: "
             << z_lower << " " << proj_center[2] << " " << z_upper);

  ASSERT(center_one[2] - z_lower <= 0.92 * radius_one and
             z_upper - center_one[2] <= 0.92 * radius_one,
         "CylindricalSide: The map has been tested only when z_lower and "
         "z_upper are sufficently far from the edge of sphere_one");

  ASSERT(center_one[2] - z_lower >= 0.01 * radius_one and
             z_upper - center_one[2] >= 0.01 * radius_one,
         "CylindricalSide: The map has been tested only when z_lower and "
         "z_upper are sufficently far from the center of sphere_one");

  if (dist_spheres + radius_two < radius_one) {
    ASSERT(center_one[2] - z_lower >= 0.2 * radius_one and
               z_upper - center_one[2] >= 0.2 * radius_one,
           "CylindricalSide: The map has been tested only when z_lower and "
           "z_upper are sufficently far from the center of sphere_one");
  }
#endif
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> CylindricalSide::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  return impl_.operator()(source_coords);
}

std::optional<std::array<double, 3>> CylindricalSide::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  return impl_.inverse(target_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalSide::jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return impl_.jacobian(source_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalSide::inv_jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return impl_.inv_jacobian(source_coords);
}

void CylindricalSide::pup(PUP::er& p) noexcept { p | impl_; }

bool operator==(const CylindricalSide& lhs,
                const CylindricalSide& rhs) noexcept {
  return lhs.impl_ == rhs.impl_;
}
bool operator!=(const CylindricalSide& lhs,
                const CylindricalSide& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                 \
  CylindricalSide::operator()(const std::array<DTYPE(data), 3>& source_coords) \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  CylindricalSide::jacobian(const std::array<DTYPE(data), 3>& source_coords)   \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  CylindricalSide::inv_jacobian(                                               \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps
