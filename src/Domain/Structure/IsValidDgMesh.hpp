// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
template <size_t Dim>
class Element;
template <size_t Dim>
class Mesh;
/// \endcond

namespace domain {
/// \brief Returns true if mesh is valid for DG on the given element
///
/// \details A Mesh is valid for DG on an Element if:
/// - the Mesh satisfies Spectral::is_valid_dg_mesh
/// - the Basis of the Mesh is appropriate for the topologies of the Element
template <size_t Dim>
bool is_valid_dg_mesh(const Mesh<Dim>& mesh, const Element<Dim>& element);
}  // namespace domain
