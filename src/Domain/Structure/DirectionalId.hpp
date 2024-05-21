// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \brief The ElementId of an Element in a given Direction.
///
/// \details Used as the key in a DirectionalIdMap
template <size_t VolumeDim>
struct DirectionalId : private ElementId<VolumeDim> {
  DirectionalId() = default;

  DirectionalId(const Direction<VolumeDim>& direction,
                const ElementId<VolumeDim>& element_id)
      : ElementId<VolumeDim>(direction, element_id) {}

  ElementId<VolumeDim> id() const { return this->without_direction(); }

  Direction<VolumeDim> direction() const {
    return ElementId<VolumeDim>::direction();
  }
  void pup(PUP::er& p) { p | static_cast<ElementId<VolumeDim>&>(*this); }
};

template <size_t VolumeDim>
size_t hash_value(const DirectionalId<VolumeDim>& id);

namespace std {
template <size_t VolumeDim>
struct hash<DirectionalId<VolumeDim>> {
  size_t operator()(const DirectionalId<VolumeDim>& id) const;
};
}  // namespace std

template <size_t VolumeDim>
bool operator==(const DirectionalId<VolumeDim>& lhs,
                const DirectionalId<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const DirectionalId<VolumeDim>& lhs,
                const DirectionalId<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
bool operator<(const DirectionalId<VolumeDim>& lhs,
               const DirectionalId<VolumeDim>& rhs);

/// Output operator for a DirectionalId.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const DirectionalId<VolumeDim>& direction);

template <size_t VolumeDim>
DirectionalId(Direction<VolumeDim> direction, ElementId<VolumeDim> id)
    -> DirectionalId<VolumeDim>;
