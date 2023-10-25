// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>

#include "Domain/Amr/Flag.hpp"

/// \cond
namespace amr {
template <size_t VolumeDim>
struct Info;
}  // namespace amr
template <size_t VolumeDim, typename T>
class DirectionalIdMap;
template <size_t VolumeDim, typename T>
class DirectionMap;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Mesh;
template <size_t VolumeDim>
class Neighbors;
/// \endcond

namespace amr {
/// \ingroup AmrGroup
/// \brief returns the neighbors and their Mesh%es of the Element with ElementId
/// `child_id`, whose parent Element is `parent` which has Info `parent_info`
/// and neighbor Info `parent_neighbor_info`
template <size_t VolumeDim>
std::pair<DirectionMap<VolumeDim, Neighbors<VolumeDim>>,
          DirectionalIdMap<VolumeDim, Mesh<VolumeDim>>>
neighbors_of_child(
    const Element<VolumeDim>& parent, const Info<VolumeDim>& parent_info,
    const std::unordered_map<ElementId<VolumeDim>, Info<VolumeDim>>&
        parent_neighbor_info,
    const ElementId<VolumeDim>& child_id);
}  // namespace amr
