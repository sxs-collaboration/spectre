// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

/// Items for initializing the DataBox%es of parallel components
namespace Initialization {

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
Mesh<Dim> element_mesh(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const ElementId<Dim>& element_id,
    const OrientationMap<Dim>& orientation = {}) noexcept {
  const auto& unoriented_extents = initial_extents[element_id.block_id()];
  Index<Dim> extents;
  for (size_t i = 0; i < Dim; ++i) {
    extents[i] = gsl::at(unoriented_extents, orientation(i));
  }
  return {extents.indices(), Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto};
}

}  // namespace Initialization
