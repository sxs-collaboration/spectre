// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Amr/Flag.hpp"

/// \cond
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
}  // namespace amr::projectors
