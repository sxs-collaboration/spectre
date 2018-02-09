// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Direction.hpp"

#include <ostream>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"

template <>
Direction<1>::Direction(const size_t dimension, const Side side) noexcept {
  ASSERT(0 == dimension or 3 == dimension or 4 == dimension or 5 == dimension,
         "dim = " << dimension
                  << ", for Direction<1> only dim = 0, 3, 4, or 5 is allowed.");
  switch (dimension) {
    case 0:
      axis_ = Axis::Xi;
      break;
    case 3:
      axis_ = Axis::X;
      break;
    case 4:
      axis_ = Axis::Y;
      break;
    case 5:
      axis_ = Axis::Z;
      break;
    default:
      ERROR("For Direction<1> only dim = 0, 3, 4, or 5 are allowed.");
  }
  side_ = side;
}

/// \cond NEVER
template <>
Direction<2>::Direction(const size_t dimension, const Side side) noexcept {
  ASSERT(0 == dimension or 1 == dimension or 3 == dimension or 4 == dimension or
             5 == dimension,
         "dim = "
             << dimension
             << ", for Direction<2> only dim = 0, 1, 3, 4, or 5 are allowed.");
  switch (dimension) {
    case 0:
      axis_ = Axis::Xi;
      break;
    case 1:
      axis_ = Axis::Eta;
      break;
    case 3:
      axis_ = Axis::X;
      break;
    case 4:
      axis_ = Axis::Y;
      break;
    case 5:
      axis_ = Axis::Z;
      break;
    default:
      ERROR("For Direction<2> only dim = 0, 1, 3, 4, or 5 are allowed.");
  }
  side_ = side;
}

template <>
Direction<3>::Direction(const size_t dimension, const Side side) noexcept {
  ASSERT(dimension < 6, "dim = " << dimension
                                 << ", for Direction<3> only dim = 0, 1, "
                                    "2, 3, 4, or 5 are allowed.");
  switch (dimension) {
    case 0:
      axis_ = Axis::Xi;
      break;
    case 1:
      axis_ = Axis::Eta;
      break;
    case 2:
      axis_ = Axis::Zeta;
      break;
    case 3:
      axis_ = Axis::X;
      break;
    case 4:
      axis_ = Axis::Y;
      break;
    case 5:
      axis_ = Axis::Z;
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
  os << direction.dimension();
  return os;
}

template std::ostream& operator<<(std::ostream&, const Direction<1>&);
template std::ostream& operator<<(std::ostream&, const Direction<2>&);
template std::ostream& operator<<(std::ostream&, const Direction<3>&);
