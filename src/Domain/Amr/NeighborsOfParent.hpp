// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <vector>

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
/// `parent_id`, that is created from its `children_elements_and_neighbor_info`
template <size_t VolumeDim>
std::pair<DirectionMap<VolumeDim, Neighbors<VolumeDim>>,
          DirectionalIdMap<VolumeDim, Mesh<VolumeDim>>>
neighbors_of_parent(
    const ElementId<VolumeDim>& parent_id,
    const std::vector<std::tuple<
        const Element<VolumeDim>&,
        const std::unordered_map<ElementId<VolumeDim>, Info<VolumeDim>>&>>&
        children_elements_and_neighbor_info);
}  // namespace amr
