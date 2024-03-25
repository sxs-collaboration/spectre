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
 * \brief Copies the neighbors' TCI decisions into
 * `subcell::Tags::NeighborTciDecisions<Dim>`
 */
template <size_t Dim>
void neighbor_tci_decision(
    gsl::not_null<db::Access*> box,
    const std::pair<
        const TimeStepId,
        DirectionalIdMap<
            Dim, std::tuple<Mesh<Dim>, Mesh<Dim - 1>, std::optional<DataVector>,
                            std::optional<DataVector>, ::TimeStepId, int>>>&
        received_temporal_id_and_data);
}  // namespace evolution::dg::subcell
