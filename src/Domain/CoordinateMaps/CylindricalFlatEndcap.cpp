// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CylindricalFlatEndcap.hpp"

#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedFlatEndcap.hpp"
#include "Domain/CoordinateMaps/FocallyLiftedMap.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::CoordinateMaps {

CylindricalFlatEndcap::CylindricalFlatEndcap(
    const std::array<double, 3>& center_one,
    const std::array<double, 3>& center_two,
    const std::array<double, 3>& proj_center, double radius_one,
    double radius_two) noexcept
    : impl_(center_two, proj_center, radius_two, false,
            FocallyLiftedInnerMaps::FlatEndcap(center_one, radius_one)) {
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
  ASSERT(center_one[2] <= center_two[2] - 1.05 * radius_two and
             center_one[2] >= center_two[2] - 6.0 * radius_two,
         "The map has only been tested for the case when the plane containing "
         "the circle is below the sphere, by an amount at least 5% of the "
         "sphere radius but not more than 5 times the sphere radius.");

  const double dist_proj = sqrt(square(center_two[0] - proj_center[0]) +
                                square(center_two[1] - proj_center[1]) +
                                square(center_two[2] - proj_center[2]));
  ASSERT(dist_proj <= 0.95 * radius_two,
         "The map has been tested only for the case "
         "when proj_center is contained inside the sphere, and no closer than "
         "95% of the way to the surface of the sphere");

  ASSERT(radius_one / radius_two <= 10.0 and radius_two / radius_one <= 10.0,
         "The map has been tested only for the case when the ratio of "
         "radius_one to radius_two is between 10 and 1/10");

  ASSERT(
      abs(center_one[0] - center_two[0]) <= radius_one + radius_two and
          abs(center_one[1] - center_two[1]) <= radius_one + radius_two,
      "The map has been tested only when the center of the sphere and the "
      "center of the circle are not too far apart in the x and y directions");
#endif
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> CylindricalFlatEndcap::operator()(
    const std::array<T, 3>& source_coords) const noexcept {
  return impl_.operator()(source_coords);
}

std::optional<std::array<double, 3>> CylindricalFlatEndcap::inverse(
    const std::array<double, 3>& target_coords) const noexcept {
  return impl_.inverse(target_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalFlatEndcap::jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return impl_.jacobian(source_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
CylindricalFlatEndcap::inv_jacobian(const std::array<T, 3>& source_coords) const
    noexcept {
  return impl_.inv_jacobian(source_coords);
}

void CylindricalFlatEndcap::pup(PUP::er& p) noexcept { p | impl_; }

bool operator==(const CylindricalFlatEndcap& lhs,
                const CylindricalFlatEndcap& rhs) noexcept {
  return lhs.impl_ == rhs.impl_;
}
bool operator!=(const CylindricalFlatEndcap& lhs,
                const CylindricalFlatEndcap& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  CylindricalFlatEndcap::operator()(                                         \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  CylindricalFlatEndcap::jacobian(                                           \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;       \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  CylindricalFlatEndcap::inv_jacobian(                                       \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps
