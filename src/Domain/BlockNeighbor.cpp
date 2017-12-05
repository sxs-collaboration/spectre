// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/BlockNeighbor.hpp"

#include <ostream>

template <size_t VolumeDim>
BlockNeighbor<VolumeDim>::BlockNeighbor(size_t id,
                                        OrientationMap<VolumeDim> orientation)
    : id_(id), orientation_(std::move(orientation)) {}

template <size_t VolumeDim>
void BlockNeighbor<VolumeDim>::pup(PUP::er& p) {
  p | id_;
  p | orientation_;
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const BlockNeighbor<VolumeDim>& block_neighbor) {
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

// Explicit instantiations
template class BlockNeighbor<1>;
template class BlockNeighbor<2>;
template class BlockNeighbor<3>;

template std::ostream& operator<<(std::ostream&, const BlockNeighbor<1>&);
template std::ostream& operator<<(std::ostream&, const BlockNeighbor<2>&);
template std::ostream& operator<<(std::ostream&, const BlockNeighbor<3>&);

template bool operator==(const BlockNeighbor<1>& lhs,
                         const BlockNeighbor<1>& rhs);
template bool operator==(const BlockNeighbor<2>& lhs,
                         const BlockNeighbor<2>& rhs);
template bool operator==(const BlockNeighbor<3>& lhs,
                         const BlockNeighbor<3>& rhs);

template bool operator!=(const BlockNeighbor<1>& lhs,
                         const BlockNeighbor<1>& rhs);
template bool operator!=(const BlockNeighbor<2>& lhs,
                         const BlockNeighbor<2>& rhs);
template bool operator!=(const BlockNeighbor<3>& lhs,
                         const BlockNeighbor<3>& rhs);
