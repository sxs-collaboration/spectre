// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <iterator>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
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
 *  FixedHashMap<
 *      maximum_number_of_neighbors(volume_dim),
 *      std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
 *      DataVector,
 *      boost::hash<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>
 * \endcode
 *
 * which holds the reconstructed `dg_packaged_data` on the face (stored in the
 * `DataVector`) for the boundary correction. A
 * `std::vector<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>`
 * holding the list of mortars that need to be reconstructed to is passed in as
 * the last argument to
 * `Metavariables::SubcellOptions::DgComputeSubcellNeighborPackagedData::apply`.
 */
template <typename Metavariables, typename DbTagsList>
void neighbor_reconstructed_face_solution(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<std::pair<
        const TimeStepId,
        FixedHashMap<
            maximum_number_of_neighbors(Metavariables::volume_dim),
            std::pair<Direction<Metavariables::volume_dim>,
                      ElementId<Metavariables::volume_dim>>,
            std::tuple<Mesh<Metavariables::volume_dim>,
                       Mesh<Metavariables::volume_dim - 1>,
                       std::optional<DataVector>, std::optional<DataVector>,
                       ::TimeStepId, int>,
            boost::hash<std::pair<Direction<Metavariables::volume_dim>,
                                  ElementId<Metavariables::volume_dim>>>>>*>
        received_temporal_id_and_data) {
  constexpr size_t volume_dim = Metavariables::volume_dim;
  db::mutate<subcell::Tags::GhostDataForReconstruction<volume_dim>,
             subcell::Tags::DataForRdmpTci>(
      [&received_temporal_id_and_data](const auto subcell_ghost_data_ptr,
                                       const auto rdmp_tci_data_ptr) {
        const size_t number_of_evolved_vars =
            rdmp_tci_data_ptr->max_variables_values.size();
        for (auto& received_mortar_data :
             received_temporal_id_and_data->second) {
          const auto& mortar_id = received_mortar_data.first;
          ASSERT(std::get<2>(received_mortar_data.second).has_value(),
                 "The subcell mortar data was not sent at TimeStepId "
                     << received_temporal_id_and_data->first
                     << " with mortar id (" << mortar_id.first << ','
                     << mortar_id.second << ")");
          const DataVector& neighbor_ghost_and_subcell_data =
              *std::get<2>(received_mortar_data.second);
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
                     << mortar_id.first
                     << " with ElementId: " << mortar_id.second);

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
  std::vector<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>
      mortars_to_reconstruct_to{};
  for (auto& received_mortar_data : received_temporal_id_and_data->second) {
    const auto& mortar_id = received_mortar_data.first;
    if (not std::get<3>(received_mortar_data.second).has_value()) {
      mortars_to_reconstruct_to.push_back(mortar_id);
    }
  }
  FixedHashMap<
      maximum_number_of_neighbors(volume_dim),
      std::pair<Direction<volume_dim>, ElementId<volume_dim>>, DataVector,
      boost::hash<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>
      neighbor_reconstructed_evolved_vars =
          Metavariables::SubcellOptions::DgComputeSubcellNeighborPackagedData::
              apply(*box, mortars_to_reconstruct_to);
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
    if (not std::get<3>(received_mortar_data.second).has_value()) {
      ASSERT(neighbor_reconstructed_evolved_vars.find(mortar_id) !=
                 neighbor_reconstructed_evolved_vars.end(),
             "Could not find mortar id (" << mortar_id.first << ','
                                          << mortar_id.second
                                          << ") in reconstructed data map.");
      std::get<3>(received_mortar_data.second) =
          std::move(neighbor_reconstructed_evolved_vars.at(mortar_id));
    }
  }
}
}  // namespace evolution::dg::subcell
