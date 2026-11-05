// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

/// \cond
template <size_t VolumeDim>
class DirectionalId;
namespace db {
class Access;
}  // namespace db
namespace evolution::dg {
template <size_t Dim>
struct BoundaryData;
}  // namespace evolution::dg
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Copies the subcell data received from a neighbor into the DataBox.
 *
 * This is invoked by elements doing DG for every neighbor across a mortar
 * without local time stepping, whether or not the element or the neighbor can
 * use subcell. Elements whose mesh does not support subcell store nothing.
 *
 * The mesh and data needed for reconstruction are copied over into
 * `subcell::Tags::MeshForGhostData` and
 * `subcell::Tags::GhostDataForReconstruction`.
 * Additionally, the max/min of the evolved variables from neighboring elements
 * that is used for the relaxed discrete maximum principle troubled-cell
 * indicator is combined with the data from the local element and stored in
 * `subcell::Tags::DataForRdmpTci`. We handle the RDMP
 * data now because it is sent in the same buffer as the data for
 * reconstruction.
 *
 * Neighbors don't send this data to elements that can never use subcell, i.e.
 * that are in or border a DG-only block (see
 * `evolution::dg::subcell::prepare_neighbor_data`), so for those elements a
 * message without ghost data is ignored. It is an error for any other element.
 */
template <size_t VolumeDim>
void receive_subcell_data_for_dg(
    gsl::not_null<db::Access*> box, const DirectionalId<VolumeDim>& mortar_id,
    const evolution::dg::BoundaryData<VolumeDim>& received_mortar_data);
}  // namespace evolution::dg::subcell
