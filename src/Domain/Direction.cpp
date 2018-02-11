// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Direction.hpp"

#include <ostream>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"

template <>
Direction<1>::Direction(const size_t logical_dimension,
                        const Side side) noexcept {
  ASSERT(0 == logical_dimension,
         "dim = " << logical_dimension
                  << ", for Direction<1> only dim = 0 is allowed.");
  axis_ = Axis::Xi;
  side_ = side;
}

/// \cond NEVER
template <>
Direction<2>::Direction(const size_t logical_dimension,
                        const Side side) noexcept {
  ASSERT(logical_dimension < 2,
         "dim = " << logical_dimension
                  << ", for Direction<2> only dim = 0 or dim = 1 are allowed.");
  axis_ = 0 == logical_dimension ? Axis::Xi : Axis ::Eta;
  side_ = side;
}

template <>
Direction<3>::Direction(const size_t logical_dimension,
                        const Side side) noexcept {
  ASSERT(logical_dimension < 3,
         "dim = " << logical_dimension
                  << ", for Direction<3> only dim = 0, dim = 1, "
                     "or dim = 2 are allowed.");
  switch (logical_dimension) {
    case 0:
      axis_ = Axis::Xi;
      break;
    case 1:
      axis_ = Axis::Eta;
      break;
    case 2:
      axis_ = Axis::Zeta;
      break;
    default:
      ERROR("For Direction<3> only dim = 0, 1, 2, 3, 4, or 5 are allowed.");
  }
  side_ = side;
}
/// \endcond

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Direction<VolumeDim>& direction) {
  if (-1.0 == direction.sign()) {
    os << "-";
  } else {
    os << "+";
  }
  os << direction.logical_dimension();
  return os;
}

template std::ostream& operator<<(std::ostream&, const Direction<1>&);
template std::ostream& operator<<(std::ostream&, const Direction<2>&);
template std::ostream& operator<<(std::ostream&, const Direction<3>&);
