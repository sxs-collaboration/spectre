// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"

namespace evolution::dg {
/*!
 * \brief The data communicated between neighber elements.
 *
 * The stored data consists of the following:
 *
 * 1. the mesh of the ghost cell data we received. This allows eliding
 *    projection when all neighboring elements are doing DG.
 * 2. the mesh of the neighboring element's face (not the mortar mesh!)
 * 3. the variables at the ghost zone cells for finite difference/volume
 *    reconstruction
 * 4. the data on the mortar needed for computing the boundary corrections (e.g.
 *    fluxes, characteristic speeds, conserved variables)
 * 5. the TimeStepId beyond which the boundary terms are no longer valid, when
 *    using local time stepping.
 * 6. the troubled cell indicator status used for determining halos around
 *    troubled cells.
 * 7. the integration order of the time-stepper
 */
template <size_t Dim>
struct BoundaryData {
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  Mesh<Dim> volume_mesh_ghost_cell_data{};
  Mesh<Dim - 1> interface_mesh{};
  std::optional<DataVector> ghost_cell_data{};
  std::optional<DataVector> boundary_correction_data{};
  ::TimeStepId validity_range{};
  int tci_status{};
  size_t integration_order{std::numeric_limits<size_t>::max()};
};

template <size_t Dim>
bool operator==(const BoundaryData<Dim>& lhs, const BoundaryData<Dim>& rhs);
template <size_t Dim>
bool operator!=(const BoundaryData<Dim>& lhs, const BoundaryData<Dim>& rhs);
template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const BoundaryData<Dim>& value);
}  // namespace evolution::dg
