// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/InterpolationTargetSpecifiedPoints.hpp"

#include <pup.h>
#include <utility>

#include "Utilities/GenerateInstantiations.hpp"

namespace intrp::OptionHolders {

template <size_t VolumeDim>
SpecifiedPoints<VolumeDim>::SpecifiedPoints(
    std::vector<std::array<double, VolumeDim>> points_in) noexcept
    : points(std::move(points_in)) {}

template <size_t VolumeDim>
void SpecifiedPoints<VolumeDim>::pup(PUP::er& p) noexcept {
  p | points;
}

template <size_t VolumeDim>
bool operator==(const SpecifiedPoints<VolumeDim>& lhs,
                const SpecifiedPoints<VolumeDim>& rhs) noexcept {
  return lhs.points == rhs.points;
}

template <size_t VolumeDim>
bool operator!=(const SpecifiedPoints<VolumeDim>& lhs,
                const SpecifiedPoints<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template struct SpecifiedPoints<DIM(data)>;                           \
  template bool operator==(const SpecifiedPoints<DIM(data)>&,           \
                           const SpecifiedPoints<DIM(data)>&) noexcept; \
  template bool operator!=(const SpecifiedPoints<DIM(data)>&,           \
                           const SpecifiedPoints<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace intrp::OptionHolders
