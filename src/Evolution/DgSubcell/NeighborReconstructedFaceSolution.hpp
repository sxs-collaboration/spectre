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
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
 * `subcell::Tags::NeighborDataForReconstructionAndRdmpTci`.
 * Additionally, the max/min of the evolved variables from neighboring elements
 * that is used for the relaxed discrete maximum principle troubled-cell
 * indicator is combined with the data from the local element and stored in
 * `subcell::Tags::NeighborDataForReconstructionAndRdmpTci`. We handle the RDMP
 * data now because it is sent in the same buffer as the data for
 * reconstruction.
 *
 * A list of all the directions that are doing subcell is created and then
 * passed to the mutator
 * `Metavariables::SubcellOptions::DgOuterCorrectionPackageData::apply`, which
 * must return a
 *
 * \code
 *  FixedHashMap<
 *      maximum_number_of_neighbors(volume_dim),
 *      std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
 *      std::vector<double>,
 *      boost::hash<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>
 * \endcode
 *
 * which holds the reconstructed `dg_packaged_data` on the face (stored in the
 * `std::vector<double>`) for the boundary correction. A
 * `std::vector<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>`
 * holding the list of mortars that need to be reconstructed to is passed in as
 * the last argument to
 * `Metavariables::SubcellOptions::DgOuterCorrectionPackageData::apply`.
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
            std::tuple<Mesh<Metavariables::volume_dim - 1>,
                       std::optional<std::vector<double>>,
                       std::optional<std::vector<double>>, ::TimeStepId>,
            boost::hash<std::pair<Direction<Metavariables::volume_dim>,
                                  ElementId<Metavariables::volume_dim>>>>>*>
        received_temporal_id_and_data) noexcept {
  constexpr size_t volume_dim = Metavariables::volume_dim;
  db::mutate<
      subcell::Tags::NeighborDataForReconstructionAndRdmpTci<volume_dim>>(
      box, [&received_temporal_id_and_data](
               const auto subcell_neighbor_data_ptr) noexcept {
        subcell::NeighborData& self_neighbor_data =
            subcell_neighbor_data_ptr->at(
                std::pair{Direction<volume_dim>::lower_xi(),
                          ElementId<volume_dim>::external_boundary_id()});
        const size_t number_of_evolved_vars =
            self_neighbor_data.max_variables_values.size();
        for (auto& received_mortar_data :
             received_temporal_id_and_data->second) {
          const auto& mortar_id = received_mortar_data.first;
          ASSERT(std::get<1>(received_mortar_data.second).has_value(),
                 "The subcell mortar data was not sent at TimeStepId "
                     << received_temporal_id_and_data->first
                     << " with mortar id (" << mortar_id.first << ','
                     << mortar_id.second << ")");
          const std::vector<double>& neighbor_ghost_and_subcell_data =
              *std::get<1>(received_mortar_data.second);
          // Compute min and max over neighbors
          const size_t offset_for_min =
              neighbor_ghost_and_subcell_data.size() - number_of_evolved_vars;
          const size_t offset_for_max = offset_for_min - number_of_evolved_vars;
          for (size_t var_index = 0; var_index < number_of_evolved_vars;
               ++var_index) {
            self_neighbor_data.max_variables_values[var_index] = std::max(
                self_neighbor_data.max_variables_values[var_index],
                neighbor_ghost_and_subcell_data[offset_for_max + var_index]);
            self_neighbor_data.min_variables_values[var_index] = std::min(
                self_neighbor_data.min_variables_values[var_index],
                neighbor_ghost_and_subcell_data[offset_for_min + var_index]);
          }

          ASSERT(subcell_neighbor_data_ptr->find(mortar_id) ==
                     subcell_neighbor_data_ptr->end(),
                 "The subcell neighbor data is already inserted. Direction: "
                     << mortar_id.first
                     << " with ElementId: " << mortar_id.second);
          // Copy over the neighbor data for reconstruction. We need this
          // since we might be doing a step unwind and the DG algorithm deletes
          // the inbox data after lifting the fluxes to the volume.
          subcell::NeighborData neighbor_data{};
          // The std::prev avoids copying over the data for the RDMP TCI, which
          // is both the maximum and minimum of each evolved variable, so
          // `2*number_of_evolved_vars` components.
          neighbor_data.data_for_reconstruction.insert(
              neighbor_data.data_for_reconstruction.end(),
              neighbor_ghost_and_subcell_data.begin(),
              std::prev(
                  neighbor_ghost_and_subcell_data.end(),
                  2 * static_cast<std::ptrdiff_t>(number_of_evolved_vars)));
          (*subcell_neighbor_data_ptr)[mortar_id] = std::move(neighbor_data);
        }
      });
  std::vector<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>
      mortars_to_reconstruct_to{};
  for (auto& received_mortar_data : received_temporal_id_and_data->second) {
    const auto& mortar_id = received_mortar_data.first;
    if (not std::get<2>(received_mortar_data.second).has_value()) {
      mortars_to_reconstruct_to.push_back(mortar_id);
    }
  }
  FixedHashMap<
      maximum_number_of_neighbors(volume_dim),
      std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
      std::vector<double>,
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
    if (not std::get<2>(received_mortar_data.second).has_value()) {
      ASSERT(neighbor_reconstructed_evolved_vars.find(mortar_id) !=
                 neighbor_reconstructed_evolved_vars.end(),
             "Could not find mortar id (" << mortar_id.first << ','
                                          << mortar_id.second
                                          << ") in reconstructed data map.");
      std::get<2>(received_mortar_data.second) =
          std::move(neighbor_reconstructed_evolved_vars.at(mortar_id));
    }
  }
}
}  // namespace evolution::dg::subcell
