// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"

/// \cond
template <size_t Dim>
struct ElementId;
template <size_t Dim>
struct OrientationMap;
/// \endcond

namespace domain {
namespace Initialization {
/// \ingroup InitializationGroup
/// \brief Construct the initial Mesh of an Element.
///
/// \details When constructing the Mesh of an Element, pass its id, and use the
/// default argument for orientation.  When constructing the mesh of a
/// neighboring Element (when constructing mortars), pass the id and orientation
/// of the neighbor.
///
/// \param initial_extents the initial extents of each Block in the Domain
/// \param element_id id of an Element or its neighbor
/// \param orientation OrientationMap of (neighboring) `element_id`
template <size_t Dim>
Mesh<Dim> create_initial_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const ElementId<Dim>& element_id,
    const OrientationMap<Dim>& orientation = {}) noexcept;
}  // namespace Initialization
}  // namespace domain
