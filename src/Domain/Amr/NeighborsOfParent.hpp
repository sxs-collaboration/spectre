// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <vector>

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
/// \brief returns the neighbors of the Element with ElementId `parent_id`,
/// that is created from its `children_elements_and_neighbor_flags`
template <size_t VolumeDim>
DirectionMap<VolumeDim, Neighbors<VolumeDim>> neighbors_of_parent(
    const ElementId<VolumeDim>& parent_id,
    const std::vector<
        std::tuple<const Element<VolumeDim>&,
                   const std::unordered_map<ElementId<VolumeDim>,
                                            std::array<Flag, VolumeDim>>&>>&
        children_elements_and_neighbor_flags);
}  // namespace amr
