// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Sets the local data from the relaxed discrete maximum principle
 * troubled-cell indicator and sends ghost zone data to neighboring elements.
 *
 * The action proceeds as follows:
 *
 * 1. Computes the maximum and minimum of each evolved variable, which is used
 *    by the relaxed discrete maximum principle troubled-cell indicator.
 * 2. Determine in which directions we have neighbors
 * 3. Slice the variables provided by GhostDataMutator to send to our neighbors
 *    for ghost zones
 * 4. Send the ghost zone data, appending the max/min for the TCI at the end of
 *    the `std::vector<double>` we are sending.
 *
 * Some notes:
 * - In the future we will need to send the cell-centered fluxes to do
 *   high-order FD without additional reconstruction being necessary.
 *
 * GlobalCache:
 * - Uses:
 *   - `ParallelComponent` proxy
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Mesh<Dim>`
 *   - `subcell::Tags::Mesh<Dim>`
 *   - `domain::Tags::Element<Dim>`
 *   - `Tags::TimeStepId`
 *   - `Tags::Next<Tags::TimeStepId>`
 *   - `subcell::Tags::ActiveGrid`
 *   - `System::variables_tag`
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::Tags::NeighborDataForReconstructionAndRdmpTci<Dim>`
 */
template <size_t Dim, typename GhostDataMutator>
struct SendDataForReconstruction {
  using inbox_tags = tmpl::list<
      evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using variables_tag = typename Metavariables::system::variables_tag;

    ASSERT(db::get<Tags::ActiveGrid>(box) == ActiveGrid::Subcell,
           "The SendDataForReconstruction action can only be called when "
           "Subcell is the active scheme.");

    db::mutate<Tags::NeighborDataForReconstructionAndRdmpTci<Dim>>(
        make_not_null(&box),
        [](const auto neighbor_data_ptr, const auto& active_vars) noexcept {
          auto [max_of_vars, min_of_vars] =
              rdmp_max_min(active_vars, {}, false);

          // Clear the previous neighbor data and add current local data
          neighbor_data_ptr->clear();
          (*neighbor_data_ptr)[std::pair{
              Direction<Dim>::lower_xi(),
              ElementId<Dim>::external_boundary_id()}] =
              NeighborData{{}, std::move(max_of_vars), std::move(min_of_vars)};
        },
        db::get<variables_tag>(box));

    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<Tags::Mesh<Dim>>(box);
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const size_t ghost_zone_size =
        Metavariables::SubcellOptions::ghost_zone_size(box);
    DirectionMap<Dim, bool> directions_to_slice{};
    for (const auto& direction_neighbors : element.neighbors()) {
      if (direction_neighbors.second.size() == 0) {
        directions_to_slice[direction_neighbors.first] = false;
      } else {
        directions_to_slice[direction_neighbors.first] = true;
      }
    }
    // Optimization note: could save a copy+allocation if we moved
    // all_sliced_data when possible before sending.
    const DirectionMap<Dim, std::vector<double>> all_sliced_data = slice_data(
        db::mutate_apply(GhostDataMutator{}, make_not_null(&box)),
        subcell_mesh.extents(), ghost_zone_size, directions_to_slice);

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const NeighborData& local_neighbor_data =
        db::get<Tags::NeighborDataForReconstructionAndRdmpTci<Dim>>(box).at(
            std::pair{Direction<Dim>::lower_xi(),
                      ElementId<Dim>::external_boundary_id()});
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const TimeStepId& next_time_step_id = [&box]() noexcept {
      if (Metavariables::local_time_stepping) {
        return db::get<::Tags::Next<::Tags::TimeStepId>>(box);
      } else {
        return db::get<::Tags::TimeStepId>(box);
      }
    }();

    // Compute and send actual variables
    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      ASSERT(neighbors_in_direction.size() == 1,
             "AMR is not yet supported when using DG-subcell. Note that this "
             "condition could be relaxed to support AMR only where the "
             "evolution is using DG without any changes to subcell.");

      for (const ElementId<Dim>& neighbor : neighbors_in_direction) {
        std::vector<double> subcell_data_to_send{};
        if (not orientation.is_aligned()) {
          std::array<size_t, Dim> slice_extents{};
          for (size_t d = 0; d < Dim; ++d) {
            gsl::at(slice_extents, d) = subcell_mesh.extents(d);
          }
          gsl::at(slice_extents, direction.dimension()) = ghost_zone_size;

          subcell_data_to_send =
              orient_variables(all_sliced_data.at(direction),
                               Index<Dim>{slice_extents}, orientation);
        } else {
          subcell_data_to_send = all_sliced_data.at(direction);
        }
        subcell_data_to_send.insert(
            subcell_data_to_send.end(),
            local_neighbor_data.max_variables_values.cbegin(),
            local_neighbor_data.max_variables_values.cend());
        subcell_data_to_send.insert(
            subcell_data_to_send.end(),
            local_neighbor_data.min_variables_values.cbegin(),
            local_neighbor_data.min_variables_values.cend());

        std::tuple<Mesh<Dim - 1>, std::optional<std::vector<double>>,
                   std::optional<std::vector<double>>, ::TimeStepId>
            data{dg_mesh.slice_away(direction.dimension()),
                 std::move(subcell_data_to_send), std::nullopt,
                 next_time_step_id};

        Parallel::receive_data<
            evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
            receiver_proxy[neighbor], time_step_id,
            std::pair{std::pair{direction_from_neighbor, element.id()},
                      std::move(data)});
      }
    }
    return {std::move(box)};
  }
};

/*!
 * \brief Receive the subcell data from our neighbor, and accumulate the data
 * from the relaxed discrete maximum principle troubled-cell indicator.
 *
 * Note:
 * - Since we only care about the min/max over all neighbors and ourself at the
 *   past time, we accumulate all data immediately into the self `NeighborData`.
 * - If the neighbor is using DG and therefore sends boundary correction data
 *   then that is added into the `evolution::dg::Tags::MortarData` tag
 * - The next `TimeStepId` is recorded, but we do not yet support local time
 *   stepping.
 * - This action will never care about what variables are sent for
 *   reconstruction. It is only responsible for receiving the data and storing
 *   it in the `NeighborData`.
 *
 * GlobalCache:
 * -Uses: nothing
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Element<Dim>`
 *   - `Tags::TimeStepId`
 *   - `domain::Tags::Mesh<Dim>`
 *   - `subcell::Tags::Mesh<Dim>`
 *   - `domain::Tags::Element<Dim>`
 *   - `Tags::Next<Tags::TimeStepId>`
 *   - `subcell::Tags::ActiveGrid`
 *   - `System::variables_tag`
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::Tags::NeighborDataForReconstructionAndRdmpTci<Dim>`
 *   - `evolution::dg::Tags::MortarData`
 *   - `evolution::dg::Tags::MortarNextTemporalId`
 */
template <size_t Dim>
struct ReceiveDataForReconstruction {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static std::tuple<db::DataBox<DbTags>&&, Parallel::AlgorithmExecution> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        not Metavariables::local_time_stepping,
        "DG-subcell does not yet support local time stepping. The "
        "reconstruction data must be sent using dense output sometimes, and "
        "not at all other times. However, the data for the RDMP TCI should be "
        "sent along with the data for reconstruction each time.");
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const auto number_of_expected_messages = element.neighbors().size();
    if (UNLIKELY(number_of_expected_messages == 0)) {
      // We have no neighbors, so just continue without doing any work
      return {std::move(box), Parallel::AlgorithmExecution::Continue};
    }

    using ::operator<<;
    using Key = std::pair<Direction<Dim>, ElementId<Dim>>;
    const auto& current_time_step_id = db::get<::Tags::TimeStepId>(box);
    std::map<TimeStepId,
             FixedHashMap<
                 maximum_number_of_neighbors(Dim), Key,
                 std::tuple<Mesh<Dim - 1>, std::optional<std::vector<double>>,
                            std::optional<std::vector<double>>, ::TimeStepId>,
                 boost::hash<Key>>>& inbox =
        tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
            Metavariables::volume_dim>>(inboxes);
    const auto& received = inbox.find(current_time_step_id);
    // Check we have at least some data from correct time, and then check that
    // we have received all data
    if (received == inbox.end() or
        received->second.size() != number_of_expected_messages) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    // Now that we have received all the data, copy it over as needed.
    FixedHashMap<maximum_number_of_neighbors(Dim), Key,
                 std::tuple<Mesh<Dim - 1>, std::optional<std::vector<double>>,
                            std::optional<std::vector<double>>, ::TimeStepId>,
                 boost::hash<Key>>
        received_data = std::move(inbox[current_time_step_id]);
    inbox.erase(current_time_step_id);
    db::mutate<Tags::NeighborDataForReconstructionAndRdmpTci<Dim>,
               evolution::dg::Tags::MortarData<Dim>,
               evolution::dg::Tags::MortarNextTemporalId<Dim>>(
        make_not_null(&box),
        [&current_time_step_id, &element, &received_data](
            const gsl::not_null<FixedHashMap<
                maximum_number_of_neighbors(Dim) + 1,
                std::pair<Direction<Dim>, ElementId<Dim>>, NeighborData,
                boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>*>
                neighbor_data_ptr,
            const gsl::not_null<std::unordered_map<
                Key, evolution::dg::MortarData<Dim>, boost::hash<Key>>*>
                mortar_data,
            const gsl::not_null<
                std::unordered_map<Key, TimeStepId, boost::hash<Key>>*>
                mortar_next_time_step_id) noexcept {
          // Get the next time step id, and also the fluxes data if the neighbor
          // is doing DG.
          for (auto& received_mortar_data : received_data) {
            const auto& mortar_id = received_mortar_data.first;
            try {
              mortar_next_time_step_id->at(mortar_id) =
                  std::get<3>(received_mortar_data.second);
            } catch (std::exception& e) {
              ERROR("Failed retrieving the MortarId: ("
                    << mortar_id.first << ',' << mortar_id.second
                    << ") from the mortar_next_time_step_id. Got exception: "
                    << e.what());
            }
            if (std::get<2>(received_mortar_data.second).has_value()) {
              mortar_data->at(mortar_id).insert_neighbor_mortar_data(
                  current_time_step_id,
                  std::get<0>(received_mortar_data.second),
                  std::move(*std::get<2>(received_mortar_data.second)));
            }
          }

          ASSERT(neighbor_data_ptr->size() == 1,
                 "Should only have one element in the neighbor data when "
                 "receiving neighbor data");
          ASSERT(
              neighbor_data_ptr->count(
                  std::pair{Direction<Dim>::lower_xi(),
                            element.id().external_boundary_id()}) == 1,
              "The self data for the RDMP TCI should have been inserted but it "
              "wasn't, despite there being one entry in the neighbor data.");
          NeighborData& self_neighbor_data = (*neighbor_data_ptr)[std::pair{
              Direction<Dim>::lower_xi(),
              ElementId<Dim>::external_boundary_id()}];
          const size_t number_of_evolved_vars =
              self_neighbor_data.max_variables_values.size();
          ASSERT(self_neighbor_data.min_variables_values.size() ==
                     number_of_evolved_vars,
                 "The number of evolved variables for which we have a maximum "
                 "and minimum should be the same, but we have "
                     << number_of_evolved_vars << " for the max and "
                     << self_neighbor_data.min_variables_values.size()
                     << " for the min.");

          for (const auto& [direction, neighbors_in_direction] :
               element.neighbors()) {
            for (const auto& neighbor : neighbors_in_direction) {
              std::pair directional_element_id{direction, neighbor};
              ASSERT(neighbor_data_ptr->count(directional_element_id) == 0,
                     "Found neighbor already inserted in direction "
                         << direction << " with ElementId " << neighbor);
              ASSERT(std::get<1>(received_data[directional_element_id])
                         .has_value(),
                     "Received subcell data message that does not contain any "
                     "actual subcell data for reconstruction.");
              // Collect the max/min of u(t^n) for the RDMP as we receive data.
              // This reduces the memory footprint.
              std::vector<double>& received_neighbor_subcell_data =
                  *std::get<1>(received_data[directional_element_id]);
              const size_t max_offset = received_neighbor_subcell_data.size() -
                                        2 * number_of_evolved_vars;
              const size_t min_offset = received_neighbor_subcell_data.size() -
                                        number_of_evolved_vars;
              for (size_t var_index = 0; var_index < number_of_evolved_vars;
                   ++var_index) {
                self_neighbor_data.max_variables_values[var_index] = std::max(
                    self_neighbor_data.max_variables_values[var_index],
                    received_neighbor_subcell_data[max_offset + var_index]);
                self_neighbor_data.min_variables_values[var_index] = std::min(
                    self_neighbor_data.min_variables_values[var_index],
                    received_neighbor_subcell_data[min_offset + var_index]);
              }
              // Copy over the ghost cell data for subcell reconstruction.
              [[maybe_unused]] const auto insert_result =
                  neighbor_data_ptr->insert(std::pair{
                      directional_element_id,
                      NeighborData{std::vector<double>{
                          received_neighbor_subcell_data.begin(),
                          std::prev(
                              received_neighbor_subcell_data.end(),
                              2 * static_cast<typename std::iterator_traits<
                                      typename std::vector<double>::iterator>::
                                                  difference_type>(
                                      number_of_evolved_vars))}}});
              ASSERT(insert_result.second,
                     "Failed to insert the neighbor data in direction "
                         << direction << " from neighbor " << neighbor);
            }
          }
        });
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};
}  // namespace evolution::dg::subcell::Actions
