// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

/// \cond
namespace amr {
enum class Flag;
template <size_t>
struct Info;
}  // namespace amr
namespace domain {
enum class Topology : uint8_t;
}  // namespace domain
template <size_t>
class Element;
template <size_t>
class ElementId;
template <size_t>
class Mesh;
/// \endcond

namespace amr::projectors {
/// Given `old_mesh` returns a new Mesh based on the refinement `flags` and
/// `topologies`
///
/// \note In dimensions that are h-refined, the Mesh is not changed by this
/// function
template <size_t Dim>
Mesh<Dim> p_refined_mesh(const Mesh<Dim>& old_mesh,
                         const std::array<amr::Flag, Dim>& flags,
                         const std::array<domain::Topology, Dim>& topologies);

/// Given the Mesh%es for a set of joining Element%s, returns the Mesh
/// for the newly created parent Element.
///
/// \details In each dimension the extent of the parent Mesh will be the
/// maximum over the extents of each child Mesh
template <size_t Dim>
Mesh<Dim> parent_mesh(const std::vector<Mesh<Dim>>& children_meshes);

/// Given the `parent_mesh`, returns the Mesh for the newly created parent
/// Element based on the refinement `flags` and `topologies`.
template <size_t Dim>
Mesh<Dim> child_mesh(const Mesh<Dim>& parent_mesh,
                     const ElementId<Dim>& child_id,
                     const std::array<amr::Flag, Dim>& flags,
                     const std::array<domain::Topology, Dim>& topologies);

/// \brief Computes the expedted new Mesh of an Element after AMR that is
/// communicated to neighboring Element%s
///
/// \details The returned Mesh will be that of either `element` or its parent or
/// children depending upon the `flags`.  If an Element is joining, the returned
/// Mesh will be that given by amr::projectors::parent_mesh.  If an Element is
/// splitting. the returned Mesh will be given by amr::projectors::child_mesh
/// for the child with Side::Upper in each split dimension. (This will be
/// incorrect for the Side::Lower child for domain::Topology::B1Radial,
/// domain::Topology::B2Radial, or domain::Topology::B3Radial; see the note.)
/// Otherwise the returned Mesh will be given by amr::projectors:p_refined_mesh
///
/// \note This function is meant to be called only by
/// amr::Actions::EvaluateRefinementCriteria and amr::Actions::UpdateAmrDecision
/// in order to communicate the expected post-refinement Mesh to neighboring
/// Elements.  For radially split ball topologies, the upper child Mesh is
/// returned because any existing neighboring elements in the radial direction
/// will have the upper child as their neighbor.
///
/// \see child_mesh parent_mesh p_refined_mesh
template <size_t Dim>
Mesh<Dim> new_mesh_to_communicate_to_neighbors(
    const Mesh<Dim>& current_mesh, const std::array<Flag, Dim>& flags,
    const Element<Dim>& element,
    const std::unordered_map<ElementId<Dim>, Info<Dim>>& neighbors_info);
}  // namespace amr::projectors
