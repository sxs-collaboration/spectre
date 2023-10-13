// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

/// \cond
namespace amr {
enum class Flag;
template <size_t>
struct Info;
}  // namespace amr
template <size_t>
class Element;
template <size_t>
class ElementId;
template <size_t>
class Mesh;
/// \endcond

namespace amr::projectors {
/// Given `old_mesh` returns a new Mesh based on the refinement `flags`
template <size_t Dim>
Mesh<Dim> mesh(const Mesh<Dim>& old_mesh,
               const std::array<amr::Flag, Dim>& flags);

/// Given the Mesh%es for a set of joining Element%s, returns the Mesh
/// for the newly created parent Element.
///
/// \details In each dimension the extent of the parent Mesh will be the
/// maximum over the extents of each child Mesh
template <size_t Dim>
Mesh<Dim> parent_mesh(const std::vector<Mesh<Dim>>& children_meshes);

/// \brief Computes the new Mesh of an Element after AMR
///
/// \details The returned Mesh will be that of either `element` or its parent or
/// children depending upon the `flags`.  If an Element is joining, the returned
/// Mesh will be that given by amr::projectors::parent_mesh; otherwise it will
/// be given by amr::projectors::mesh
template <size_t Dim>
Mesh<Dim> new_mesh(
    const Mesh<Dim>& current_mesh, const std::array<Flag, Dim>& flags,
    const Element<Dim>& element,
    const std::unordered_map<ElementId<Dim>, Info<Dim>>& neighbors_info);
}  // namespace amr::projectors
