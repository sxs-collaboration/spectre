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
template <size_t VolumeDim, typename IdType>
class Neighbors;
/// \endcond

namespace amr {
/*!
 * \ingroup AmrGroup
 * \brief Determine the new neighbors of an element during AMR, and the
 * neighbors' meshes.
 *
 * Can be used for both h-refinement and p-refinement.
 *
 * \param parent The parent element that is being refined. For h-refinement,
 * this is the element that is being split into children. For p-refinement,
 * this is the element that is being increased in resolution.
 * \param parent_info The AMR info of the parent element.
 * \param parent_neighbor_info The AMR info of the parent element's neighbors.
 * \param child_id The ID of the child element that is being created. For
 * h-refinement, this is the ID of the new child element. For p-refinement,
 * this is the same as the ID of the parent element.
 */
template <size_t VolumeDim>
std::pair<DirectionMap<VolumeDim, Neighbors<VolumeDim, ElementId<VolumeDim>>>,
          DirectionalIdMap<VolumeDim, Mesh<VolumeDim>>>
neighbors_of_child(
    const Element<VolumeDim>& parent, const Info<VolumeDim>& parent_info,
    const std::unordered_map<ElementId<VolumeDim>, Info<VolumeDim>>&
        parent_neighbor_info,
    const ElementId<VolumeDim>& child_id);
}  // namespace amr
