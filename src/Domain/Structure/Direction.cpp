// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/Direction.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>

#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim>
Direction<VolumeDim> Direction<VolumeDim>::self() {
  return Direction<VolumeDim>{Axis::Xi, Side::Self};
}

template <size_t VolumeDim>
Direction<VolumeDim>::Direction(Axis axis, Side side)
    : bit_field_(static_cast<uint8_t>(axis) bitor static_cast<uint8_t>(side)) {
  ASSERT(bit_field_ < 0b10000, "Direction is too large.");
}

template <size_t VolumeDim>
void Direction<VolumeDim>::pup(PUP::er& p) {
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  const uint8_t current_version = 1 << 6;
  if (p.isUnpacking()) {
    uint8_t data = 0;
    p | data;
    const uint8_t version_read = (0b11000000 bitand data) >> 6;
    if (LIKELY(version_read > 0)) {
      bit_field_ = data bitand 0b00001111;
    } else {
      // Read next 3 bytes.
      p | data;
      p | data;
      p | data;
      // Read "bottom" of version
      uint32_t old_version{0};
      p | old_version;
      if (UNLIKELY(old_version > 0)) {
        ERROR(
            "Incompatible version format for Direction. Expected to receive "
            "version 0 but got "
            << old_version);
      }
      int read_axis{0};
      p | read_axis;
      if (UNLIKELY(read_axis < 0)) {
        ERROR("Expected a non-negative axis value but read " << read_axis);
      }
      int read_side{0};
      p | read_side;
      if (UNLIKELY(read_side < 0)) {
        ERROR("Expected a non-negative side value but read " << read_side);
      }
      // add 1 to account for the added Uninitialized
      const auto side = static_cast<uint8_t>(read_side + 1)
                        << detail::side_shift;
      bit_field_ = 0;
      bit_field_ = static_cast<uint8_t>(read_axis) bitor side;
    }
  } else {
    uint8_t data = current_version bitor bit_field_;
    p | data;
  }
}

template <size_t VolumeDim>
Direction<VolumeDim>::Direction(const size_t dimension, const Side side)
    : Direction(static_cast<Axis>(static_cast<uint8_t>(dimension)), side) {
  ASSERT(dimension < VolumeDim,
         "dim = " << dimension << ", but must be less than " << VolumeDim);
}

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

template <size_t VolumeDim>
bool operator<(const Direction<VolumeDim>& lhs,
               const Direction<VolumeDim>& rhs) {
  if (lhs.axis() != rhs.axis()) {
    return lhs.axis() < rhs.axis();
  }
  return lhs.side() < rhs.side();
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                        \
  template std::ostream& operator<<(std::ostream&,                    \
                                    const Direction<GET_DIM(data)>&); \
  template bool operator<(const Direction<GET_DIM(data)>& lhs,        \
                          const Direction<GET_DIM(data)>& rhs);       \
  template void Direction<GET_DIM(data)>::pup(PUP::er&);              \
  template Direction<GET_DIM(data)>::Direction(Axis, Side);           \
  template Direction<GET_DIM(data)>::Direction(size_t, Side);         \
  template Direction<GET_DIM(data)> Direction<GET_DIM(data)>::self();

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
