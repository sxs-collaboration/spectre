// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
/*!
 * \brief Copies the neighbors' TCI decisions into
 * `subcell::Tags::NeighborTciDecisions<Dim>`
 */
template <size_t Dim, typename DbTagsList>
void neighbor_tci_decision(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const std::pair<
        const TimeStepId,
        DirectionalIdMap<
            Dim, std::tuple<Mesh<Dim>, Mesh<Dim - 1>, std::optional<DataVector>,
                            std::optional<DataVector>, ::TimeStepId, int>>>&
        received_temporal_id_and_data) {
  db::mutate<subcell::Tags::NeighborTciDecisions<Dim>>(
      [&received_temporal_id_and_data](const auto neighbor_tci_decisions_ptr) {
        for (const auto& [directional_element_id, neighbor_data] :
             received_temporal_id_and_data.second) {
          ASSERT(neighbor_tci_decisions_ptr->contains(directional_element_id),
                 "The NeighborTciDecisions tag does not contain the neighbor "
                     << directional_element_id);
          neighbor_tci_decisions_ptr->at(directional_element_id) =
              std::get<5>(neighbor_data);
        }
      },
      box);
}
}  // namespace evolution::dg::subcell
