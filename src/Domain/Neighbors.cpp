// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Neighbors.hpp"

#include <ostream>
#include <pup.h>  // IWYU pragma: keep

#include "Domain/ElementId.hpp"      // IWYU pragma: keep
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace domain {
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
  ::operator<<(os << "Ids = ", n.ids())
      << "; orientation = " << n.orientation();
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
void Neighbors<VolumeDim>::pup(PUP::er& p) noexcept {
  p | ids_;
  p | orientation_;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template class Neighbors<DIM(data)>;                                  \
  template std::ostream& operator<<(std::ostream& os,                   \
                                    const Neighbors<DIM(data)>& block); \
  template bool operator==(const Neighbors<DIM(data)>& lhs,             \
                           const Neighbors<DIM(data)>& rhs) noexcept;   \
  template bool operator!=(const Neighbors<DIM(data)>& lhs,             \
                           const Neighbors<DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace domain
