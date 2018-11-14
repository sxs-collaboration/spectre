// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/InterpolationTargetLineSegment.hpp"

#include "Utilities/GenerateInstantiations.hpp"

namespace intrp {
namespace OptionHolders {

template <size_t VolumeDim>
LineSegment<VolumeDim>::LineSegment(std::array<double, VolumeDim> begin_in,
                                    std::array<double, VolumeDim> end_in,
                                    size_t number_of_points_in) noexcept
    : begin(std::move(begin_in)),  // NOLINT
      end(std::move(end_in)),      // NOLINT
      number_of_points(number_of_points_in) {}
// above NOLINT for std::move of trivially copyable type.

template <size_t VolumeDim>
void LineSegment<VolumeDim>::pup(PUP::er& p) noexcept {
  p | begin;
  p | end;
  p | number_of_points;
}

template <size_t VolumeDim>
bool operator==(const LineSegment<VolumeDim>& lhs,
                const LineSegment<VolumeDim>& rhs) noexcept {
  return lhs.begin == rhs.begin and lhs.end == rhs.end and
         lhs.number_of_points == rhs.number_of_points;
}

template <size_t VolumeDim>
bool operator!=(const LineSegment<VolumeDim>& lhs,
                const LineSegment<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template struct LineSegment<DIM(data)>;                           \
  template bool operator==(const LineSegment<DIM(data)>&,           \
                           const LineSegment<DIM(data)>&) noexcept; \
  template bool operator!=(const LineSegment<DIM(data)>&,           \
                           const LineSegment<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace OptionHolders
}  // namespace intrp
