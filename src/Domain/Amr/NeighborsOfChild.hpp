// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>

#include "Domain/Amr/Flag.hpp"

/// \cond
template <size_t VolumeDim, typename T>
class DirectionMap;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Neighbors;
/// \endcond

namespace amr {
/// \ingroup AmrGroup
/// \brief returns the neighbors of the Element with ElementId `child_id`,
/// whose parent Element is `parent` which has refinement flags `parent_flags`
/// and neighbor flags `parent_neighbor_flags`
template <size_t VolumeDim>
DirectionMap<VolumeDim, Neighbors<VolumeDim>> neighbors_of_child(
    const Element<VolumeDim>& parent,
    const std::array<Flag, VolumeDim>& parent_flags,
    const std::unordered_map<ElementId<VolumeDim>, std::array<Flag, VolumeDim>>&
        parent_neighbor_flags,
    const ElementId<VolumeDim>& child_id);
}  // namespace amr
