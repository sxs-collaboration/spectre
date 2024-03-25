// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/NeighborTciDecision.hpp"

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
template <size_t Dim>
void neighbor_tci_decision(
    const gsl::not_null<db::Access*> box,
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

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template void neighbor_tci_decision(                                         \
      gsl::not_null<db::Access*> box,                                          \
      const std::pair<                                                         \
          const TimeStepId,                                                    \
          DirectionalIdMap<                                                    \
              DIM(data),                                                       \
              std::tuple<Mesh<DIM(data)>, Mesh<DIM(data) - 1>,                 \
                         std::optional<DataVector>, std::optional<DataVector>, \
                         ::TimeStepId, int>>>& received_temporal_id_and_data);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell
