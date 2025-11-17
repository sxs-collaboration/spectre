// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

/// \cond
template <size_t Dim, typename T>
class DirectionalIdMap;
class TimeStepId;
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
 * \brief Invoked in directions where the neighbor is doing subcell, this
 * function copies received subcell data into the DataBox.
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
 */
template <size_t VolumeDim>
void receive_subcell_data_for_dg(
    gsl::not_null<db::Access*> box,
    const std::pair<
        TimeStepId,
        DirectionalIdMap<VolumeDim, evolution::dg::BoundaryData<VolumeDim>>>&
        received_temporal_id_and_data);
}  // namespace evolution::dg::subcell
