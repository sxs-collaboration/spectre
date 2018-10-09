// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/BlockNeighbor.hpp"

#include <ostream>

#include "Utilities/GenerateInstantiations.hpp"

template <size_t VolumeDim>
BlockNeighbor<VolumeDim>::BlockNeighbor(
    size_t id, OrientationMap<VolumeDim> orientation) noexcept
    : id_(id), orientation_(std::move(orientation)) {}

template <size_t VolumeDim>
void BlockNeighbor<VolumeDim>::pup(PUP::er& p) noexcept {
  p | id_;
  p | orientation_;
}

template <size_t VolumeDim>
std::ostream& operator<<(
    std::ostream& os, const BlockNeighbor<VolumeDim>& block_neighbor) noexcept {
  os << "Id = " << block_neighbor.id()
     << "; orientation = " << block_neighbor.orientation();
  return os;
}

template <size_t VolumeDim>
bool operator==(const BlockNeighbor<VolumeDim>& lhs,
                const BlockNeighbor<VolumeDim>& rhs) noexcept {
  return lhs.id() == rhs.id() and lhs.orientation() == rhs.orientation();
}

template <size_t VolumeDim>
bool operator!=(const BlockNeighbor<VolumeDim>& lhs,
                const BlockNeighbor<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                \
  template class BlockNeighbor<GET_DIM(data)>;                                \
  template std::ostream& operator<<(                                          \
      std::ostream& os, const BlockNeighbor<GET_DIM(data)>& block);           \
  template bool operator==(const BlockNeighbor<GET_DIM(data)>& lhs,           \
                           const BlockNeighbor<GET_DIM(data)>& rhs) noexcept; \
  template bool operator!=(const BlockNeighbor<GET_DIM(data)>& lhs,           \
                           const BlockNeighbor<GET_DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
