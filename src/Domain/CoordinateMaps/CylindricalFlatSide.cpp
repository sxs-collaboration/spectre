// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CylindricalFlatSide.hpp"

#include <optional>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedFlatSide.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::CoordinateMaps {

CylindricalFlatSide::CylindricalFlatSide(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two,
    const std::array<double, 3>& proj_center, const double inner_radius,
    const double outer_radius, const double radius_two) noexcept
    : impl_(center_two, proj_center, radius_two, false,
            FocallyLiftedInnerMaps::FlatSide(center_one, inner_radius,
                                             outer_radius)) {
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
  //    that we don't expect to be used.
  ASSERT(center_one[2] < center_two[2] - radius_two * 1.05,
         "CylindricalFlatSide: The map has only been tested for the case "
         "when the sphere is contained at larger z-coordinate than the "
         "plane that contains the annulus, and the sphere is not too "
         "close to the annulus.");
  const double dist_proj = sqrt(square(center_two[0] - proj_center[0]) +
                                square(center_two[1] - proj_center[1]) +
                                square(center_two[2] - proj_center[2]));
  ASSERT(dist_proj < 0.95 * radius_two,
         "CylindricalFlatSide: The map has been tested only for the case when "
         "proj_center is contained inside the sphere and not too close to the "
         "surface of the sphere");
  const double dist_xy_annulus = sqrt(square(center_two[0] - center_one[0]) +
                                      square(center_two[1] - center_one[1]));
  ASSERT(dist_xy_annulus < radius_two,
         "CylindricalFlatSide: The map has been tested only for the case when "
         "the center of the annulus is contained in the projection of the "
         "sphere onto the z-axis");
  const double dist_annulus_proj = sqrt(square(center_one[0] - proj_center[0]) +
                                        square(center_one[1] - proj_center[1]) +
                                        square(center_one[2] - proj_center[2]));
  ASSERT(outer_radius > 0.05 * dist_annulus_proj,
         "CylindricalFlatSide: The map has been tested only for the case when "
         "the annulus is not too small");
  ASSERT(inner_radius < 0.95 * outer_radius,
         "CylindricalFlatSide: The map has been tested only for the case when "
         "the annulus is not too thin");
  const double min_inner_radius_fac =
      std::max(0.05, dist_annulus_proj * 0.01 / outer_radius);
  ASSERT(inner_radius > min_inner_radius_fac * outer_radius,
         "CylindricalFlatSide: The map has been tested only for the case when "
         "the inner radius is not too small");
#endif
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> CylindricalFlatSide::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  return impl_.operator()(source_coords);
}

std::optional<std::array<double, 3>> CylindricalFlatSide::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  return impl_.inverse(target_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalFlatSide::jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  return impl_.jacobian(source_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalFlatSide::inv_jacobian(
    const std::array<T, 3>& source_coords) const noexcept {
  return impl_.inv_jacobian(source_coords);
}

void CylindricalFlatSide::pup(PUP::er& p) noexcept { p | impl_; }

bool operator==(const CylindricalFlatSide& lhs,
                const CylindricalFlatSide& rhs) noexcept {
  return lhs.impl_ == rhs.impl_;
}
bool operator!=(const CylindricalFlatSide& lhs,
                const CylindricalFlatSide& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  CylindricalFlatSide::operator()(                                           \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  CylindricalFlatSide::jacobian(                                             \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  CylindricalFlatSide::inv_jacobian(                                         \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond

}  // namespace domain::CoordinateMaps
