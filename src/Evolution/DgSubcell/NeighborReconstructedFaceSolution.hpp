// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
/*!
 * \brief Invoked in directions where the neighbor is doing subcell, this
 * function computes the neighbor data on the mortar via reconstruction on
 * nearest neighbor subcells.
 *
 *
 * The data needed for reconstruction is copied over into
 * `subcell::Tags::GhostDataForReconstruction`.
 * Additionally, the max/min of the evolved variables from neighboring elements
 * that is used for the relaxed discrete maximum principle troubled-cell
 * indicator is combined with the data from the local element and stored in
 * `subcell::Tags::DataForRdmpTci`. We handle the RDMP
 * data now because it is sent in the same buffer as the data for
 * reconstruction.
 *
 * A list of all the directions that are doing subcell is created and then
 * passed to the mutator
 * `Metavariables::SubcellOptions::DgComputeSubcellNeighborPackagedData::apply`,
 * which must return a
 *
 * \code
 *  DirectionalIdMap<volume_dim, DataVector>
 * \endcode
 *
 * which holds the reconstructed `dg_packaged_data` on the face (stored in the
 * `DataVector`) for the boundary correction. A
 * `std::vector<DirectionalId<volume_dim>>`
 * holding the list of mortars that need to be reconstructed to is passed in as
 * the last argument to
 * `Metavariables::SubcellOptions::DgComputeSubcellNeighborPackagedData::apply`.
 */
template <size_t VolumeDim, typename DgComputeSubcellNeighborPackagedData>
void neighbor_reconstructed_face_solution(
    gsl::not_null<db::Access*> box,
    gsl::not_null<std::pair<
        const TimeStepId,
        DirectionalIdMap<
            VolumeDim,
            std::tuple<Mesh<VolumeDim>, Mesh<VolumeDim - 1>,
                       std::optional<DataVector>, std::optional<DataVector>,
                       ::TimeStepId, int>>>*>
        received_temporal_id_and_data);
}  // namespace evolution::dg::subcell
