// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/InterpolatedBoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"

/// \cond
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg {
/*!
 * \brief The data communicated between neighber elements.
 *
 * The stored data consists of the following:
 *
 * 1. the volume mesh of the element.
 * 2. the volume mesh corresponding to the ghost cell data. This allows eliding
 *    projection when all neighboring elements are doing DG.
 * 3. the mortar mesh of the data on the mortar
 * 4. the variables at the ghost zone cells for finite difference/volume
 *    reconstruction
 * 5. the data on the mortar needed for computing the boundary corrections (e.g.
 *    fluxes, characteristic speeds, conserved variables)
 * 6. the TimeStepId beyond which the boundary terms are no longer valid, when
 *    using local time stepping.
 * 7. the troubled cell indicator status used for determining halos around
 *    troubled cells.
 * 8. the integration order of the time-stepper
 * 9. the InterpolatedBoundaryData sent by a non-conforming Element that
 *    interpolates its data to a subset of the points of the Element receiving
 *    this BoundaryData
 */
template <size_t Dim>
struct BoundaryData {
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  Mesh<Dim> volume_mesh{};
  std::optional<Mesh<Dim>> volume_mesh_ghost_cell_data{};
  std::optional<Mesh<Dim - 1>> boundary_correction_mesh{};
  std::optional<DataVector> ghost_cell_data{};
  std::optional<DataVector> boundary_correction_data{};
  ::TimeStepId validity_range{};
  int tci_status{};
  size_t integration_order{std::numeric_limits<size_t>::max()};
  std::optional<InterpolatedBoundaryData<Dim>> interpolated_boundary_data{};
};

template <size_t Dim>
bool operator==(const BoundaryData<Dim>& lhs, const BoundaryData<Dim>& rhs);
template <size_t Dim>
bool operator!=(const BoundaryData<Dim>& lhs, const BoundaryData<Dim>& rhs);
template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const BoundaryData<Dim>& value);

/*!
 * \brief Merge DG boundary correction data into an existing
 * BoundaryData object.
 *
 * In a 2-send implementation, we can receive DG boundary correction
 * data at a time for which we have already received ghost cell data.
 * This function sanity checks that the data we already have is the
 * ghost cells and then copes in the DG data.
 *
 * \note We do not currently use a 2-send implementation.  We
 * generally find that the number of communications is more important
 * than the size of each communication, and so a single communication
 * per time/sub step is preferred.
 */
template <size_t Dim>
void merge_boundary_data(gsl::not_null<BoundaryData<Dim>*> destination,
                         BoundaryData<Dim> source);
}  // namespace evolution::dg
