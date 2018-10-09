// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Direction.hpp"

#include <ostream>

#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

/// \cond
template <size_t VolumeDim>
void Direction<VolumeDim>::pup(PUP::er& p) noexcept {
  p | axis_;
  p | side_;
}

template <>
Direction<1>::Direction(const size_t dimension, const Side side) noexcept {
  ASSERT(
      0 == dimension,
      "dim = " << dimension << ", for Direction<1> only dim = 0 is allowed.");
  axis_ = Axis::Xi;
  side_ = side;
}

template <>
Direction<2>::Direction(const size_t dimension, const Side side) noexcept {
  ASSERT(0 == dimension or 1 == dimension,
         "dim = " << dimension
                  << ", for Direction<2> only dim = 0 or dim = 1 are allowed.");
  axis_ = 0 == dimension ? Axis::Xi : Axis::Eta;
  side_ = side;
}

template <>
Direction<3>::Direction(const size_t dimension, const Side side) noexcept {
  ASSERT(0 == dimension or 1 == dimension or 2 == dimension,
         "dim = " << dimension << ", for Direction<3> only dim = 0, dim = 1, "
                                  "or dim = 2 are allowed.");
  if (0 == dimension) {
    axis_ = Axis::Xi;
  }
  if (1 == dimension) {
    axis_ = Axis::Eta;
  }
  if (2 == dimension) {
    axis_ = Axis::Zeta;
  }
  side_ = side;
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Direction<VolumeDim>& direction) noexcept {
  if (-1.0 == direction.sign()) {
    os << "-";
  } else {
    os << "+";
  }
  os << direction.dimension();
  return os;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template std::ostream& operator<<(std::ostream&,                             \
                                    const Direction<GET_DIM(data)>&) noexcept; \
  template void Direction<GET_DIM(data)>::pup(PUP::er&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
/// \endcond
