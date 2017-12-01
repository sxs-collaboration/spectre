// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Neighbors.hpp"

#include <charm.h>
#include <ostream>

#include "Domain/ElementId.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/StdHelpers.hpp"

template <size_t VolumeDim>
Neighbors<VolumeDim>::Neighbors(std::unordered_set<ElementId<VolumeDim>> ids,
                                OrientationMap<VolumeDim> orientation)
    : ids_(std::move(ids)), orientation_(std::move(orientation)) {}

template <size_t VolumeDim>
void Neighbors<VolumeDim>::add_ids(
    const std::unordered_set<ElementId<VolumeDim>>& additional_ids) {
  for (const auto& id : additional_ids) {
    ids_.insert(id);
  }
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Neighbors<VolumeDim>& n) {
  os << "Ids = " << n.ids() << "; orientation = " << n.orientation();
  return os;
}

template <size_t VolumeDim>
bool operator==(const Neighbors<VolumeDim>& lhs,
                const Neighbors<VolumeDim>& rhs) noexcept {
  return (lhs.ids() == rhs.ids() and lhs.orientation() == rhs.orientation());
}

template <size_t VolumeDim>
bool operator!=(const Neighbors<VolumeDim>& lhs,
                const Neighbors<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
void Neighbors<VolumeDim>::pup(PUP::er& p) {
  p | ids_;
  p | orientation_;
}

// Explicit instantiations
template class Neighbors<1>;
template class Neighbors<2>;
template class Neighbors<3>;

template std::ostream& operator<<(std::ostream&, const Neighbors<1>&);
template std::ostream& operator<<(std::ostream&, const Neighbors<2>&);
template std::ostream& operator<<(std::ostream&, const Neighbors<3>&);

template bool operator==(const Neighbors<1>&, const Neighbors<1>&);
template bool operator==(const Neighbors<2>&, const Neighbors<2>&);
template bool operator==(const Neighbors<3>&, const Neighbors<3>&);

template bool operator!=(const Neighbors<1>&, const Neighbors<1>&);
template bool operator!=(const Neighbors<2>&, const Neighbors<2>&);
template bool operator!=(const Neighbors<3>&, const Neighbors<3>&);
