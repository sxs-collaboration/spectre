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

template <size_t VolumeDim>
struct DirectionId {
  Direction<VolumeDim> direction;
  ElementId<VolumeDim> id;

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <size_t VolumeDim>
size_t hash_value(const DirectionId<VolumeDim>& id);

namespace std {
template <size_t VolumeDim>
struct hash<DirectionId<VolumeDim>> {
  size_t operator()(const DirectionId<VolumeDim>& id) const;
};
}  // namespace std

template <size_t VolumeDim>
bool operator==(const DirectionId<VolumeDim>& lhs,
                const DirectionId<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const DirectionId<VolumeDim>& lhs,
                const DirectionId<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
bool operator<(const DirectionId<VolumeDim>& lhs,
               const DirectionId<VolumeDim>& rhs);

/// Output operator for a DirectionId.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const DirectionId<VolumeDim>& direction);
