// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace Spectral {
/// \brief Returns true if mesh is valid for DG
///
/// \details A mesh is valid if it:
/// - does not have a Basis::FiniteDifference
/// - has a consistent Basis and Quadrature (based on topology),
/// - has valid extents for the Basis and Quadrature.
/// In addition:
/// - for Quadrature::Equiangular (as part of periodic angular topologoy such
///   as S1, S2, B2, or B3), we require that the extent is odd in that
///   dimension
/// - For multidimensional topologies, we enforce that their extents
///   in the coupled dimensions are kept in sync in order to resolve the
///   highest angular mode
template <size_t Dim>
bool is_valid_dg_mesh(const Mesh<Dim>& mesh);
}  // namespace Spectral
