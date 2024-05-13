// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
template <size_t VolumeDim, typename DgComputeSubcellNeighborPackagedData>
void neighbor_reconstructed_face_solution(
    const gsl::not_null<db::Access*> box,
    const gsl::not_null<std::pair<
        const TimeStepId,
        DirectionalIdMap<VolumeDim, evolution::dg::BoundaryData<VolumeDim>>>*>
        received_temporal_id_and_data) {
  db::mutate<subcell::Tags::GhostDataForReconstruction<VolumeDim>,
             subcell::Tags::DataForRdmpTci>(
      [&received_temporal_id_and_data](const auto subcell_ghost_data_ptr,
                                       const auto rdmp_tci_data_ptr) {
        const size_t number_of_evolved_vars =
            rdmp_tci_data_ptr->max_variables_values.size();
        for (auto& received_mortar_data :
             received_temporal_id_and_data->second) {
          const auto& mortar_id = received_mortar_data.first;
          ASSERT(received_mortar_data.second.ghost_cell_data.has_value(),
                 "The subcell mortar data was not sent at TimeStepId "
                     << received_temporal_id_and_data->first
                     << " with mortar id " << mortar_id);
          const DataVector& neighbor_ghost_and_subcell_data =
              received_mortar_data.second.ghost_cell_data.value();
          // Compute min and max over neighbors
          const size_t offset_for_min =
              neighbor_ghost_and_subcell_data.size() - number_of_evolved_vars;
          const size_t offset_for_max = offset_for_min - number_of_evolved_vars;
          for (size_t var_index = 0; var_index < number_of_evolved_vars;
               ++var_index) {
            rdmp_tci_data_ptr->max_variables_values[var_index] = std::max(
                rdmp_tci_data_ptr->max_variables_values[var_index],
                neighbor_ghost_and_subcell_data[offset_for_max + var_index]);
            rdmp_tci_data_ptr->min_variables_values[var_index] = std::min(
                rdmp_tci_data_ptr->min_variables_values[var_index],
                neighbor_ghost_and_subcell_data[offset_for_min + var_index]);
          }

          ASSERT(subcell_ghost_data_ptr->find(mortar_id) ==
                     subcell_ghost_data_ptr->end(),
                 "The subcell neighbor data is already inserted. Direction: "
                     << mortar_id.direction()
                     << " with ElementId: " << mortar_id.id());

          (*subcell_ghost_data_ptr)[mortar_id] = GhostData{1};
          GhostData& all_ghost_data = subcell_ghost_data_ptr->at(mortar_id);
          DataVector& neighbor_data =
              all_ghost_data.neighbor_ghost_data_for_reconstruction();
          neighbor_data.destructive_resize(
              neighbor_ghost_and_subcell_data.size() -
              2 * number_of_evolved_vars);

          // Copy over the neighbor data for reconstruction. We need this
          // since we might be doing a step unwind and the DG algorithm deletes
          // the inbox data after lifting the fluxes to the volume.
          // The std::prev avoids copying over the data for the RDMP TCI, which
          // is both the maximum and minimum of each evolved variable, so
          // `2*number_of_evolved_vars` components.
          std::copy(neighbor_ghost_and_subcell_data.begin(),
                    std::prev(neighbor_ghost_and_subcell_data.end(),
                              2 * static_cast<std::ptrdiff_t>(
                                      number_of_evolved_vars)),
                    neighbor_data.begin());
        }
      },
      box);
  std::vector<DirectionalId<VolumeDim>> mortars_to_reconstruct_to{};
  for (auto& received_mortar_data : received_temporal_id_and_data->second) {
    const auto& mortar_id = received_mortar_data.first;
    if (not received_mortar_data.second.boundary_correction_data.has_value()) {
      mortars_to_reconstruct_to.push_back(mortar_id);
    }
  }
  DirectionalIdMap<VolumeDim, DataVector> neighbor_reconstructed_evolved_vars =
      DgComputeSubcellNeighborPackagedData::apply(*box,
                                                  mortars_to_reconstruct_to);
  ASSERT(neighbor_reconstructed_evolved_vars.size() ==
             mortars_to_reconstruct_to.size(),
         "Should have reconstructed "
             << mortars_to_reconstruct_to.size() << " sides but reconstructed "
             << neighbor_reconstructed_evolved_vars.size() << " sides.");
  // Now copy over the packaged data _into_ the inbox in order to avoid having
  // to make other changes to the DG algorithm (code in
  // src/Evolution/DiscontinuousGalerkin)
  for (auto& received_mortar_data : received_temporal_id_and_data->second) {
    const auto& mortar_id = received_mortar_data.first;
    if (not received_mortar_data.second.boundary_correction_data.has_value()) {
      ASSERT(neighbor_reconstructed_evolved_vars.find(mortar_id) !=
                 neighbor_reconstructed_evolved_vars.end(),
             "Could not find mortar id " << mortar_id
                                         << " in reconstructed data map.");
      received_mortar_data.second.boundary_correction_data =
          std::move(neighbor_reconstructed_evolved_vars.at(mortar_id));
    }
  }
}
}  // namespace evolution::dg::subcell
