// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Structure/TrimMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/NeighborRdmpAndVolumeData.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Reconstructor.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Sets the local data from the relaxed discrete maximum principle
 * troubled-cell indicator and sends ghost zone data to neighboring elements.
 *
 * The action proceeds as follows:
 *
 * 1. Determine in which directions we have neighbors
 * 2. Slice the variables provided by GhostDataMutator to send to our neighbors
 *    for ghost zones
 * 3. Send the ghost zone data, appending the max/min for the TCI at the end of
 *    the `DataVector` we are sending.
 *
 * \warning This assumes the RDMP TCI data in the DataBox has been set, it does
 * not calculate it automatically. The reason is this way we can only calculate
 * the RDMP data when it's needed since computing it can be pretty expensive.
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
 *   - `subcell::Tags::DataForRdmpTci`
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::Tags::GhostDataForReconstruction<Dim>`
 */
template <size_t Dim, typename GhostDataMutator, bool LocalTimeStepping>
struct SendDataForReconstruction {
  using inbox_tags = tmpl::list<
      evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        not LocalTimeStepping,
        "DG-subcell does not yet support local time stepping. The "
        "reconstruction data must be sent using dense output sometimes, and "
        "not at all other times. However, the data for the RDMP TCI should be "
        "sent along with the data for reconstruction each time.");

    ASSERT(db::get<Tags::ActiveGrid>(box) == ActiveGrid::Subcell,
           "The SendDataForReconstruction action can only be called when "
           "Subcell is the active scheme.");
    using flux_variables = typename Metavariables::system::flux_variables;

    db::mutate<Tags::GhostDataForReconstruction<Dim>>(
        [](const auto ghost_data_ptr) {
          // Clear the previous neighbor data and add current local data
          ghost_data_ptr->clear();
        },
        make_not_null(&box));

    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<Tags::Mesh<Dim>>(box);
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const size_t ghost_zone_size =
        db::get<evolution::dg::subcell::Tags::Reconstructor>(box)
            .ghost_zone_size();

    // Optimization note: could save a copy+allocation if we moved
    // all_sliced_data when possible before sending.
    //
    // Note: RDMP size doesn't help here since we need to slice data after
    // anyway, so no way to save an allocation through that.
    const auto& cell_centered_flux =
        db::get<Tags::CellCenteredFlux<flux_variables, Dim>>(box);
    DataVector volume_data_to_slice = db::mutate_apply(
        GhostDataMutator{}, make_not_null(&box),
        cell_centered_flux.has_value() ? cell_centered_flux.value().size()
                                       : 0_st);
    if (cell_centered_flux.has_value()) {
      std::copy(
          cell_centered_flux.value().data(),
          std::next(
              cell_centered_flux.value().data(),
              static_cast<std::ptrdiff_t>(cell_centered_flux.value().size())),
          std::next(
              volume_data_to_slice.data(),
              static_cast<std::ptrdiff_t>(volume_data_to_slice.size() -
                                          cell_centered_flux.value().size())));
    }
    const DirectionMap<Dim, DataVector> all_sliced_data = slice_data(
        volume_data_to_slice, subcell_mesh.extents(), ghost_zone_size,
        element.internal_boundaries(), 0,
        db::get<
            evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>>(
            box));

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const RdmpTciData& rdmp_tci_data = db::get<Tags::DataForRdmpTci>(box);
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    const TimeStepId& next_time_step_id = [&box]() {
      if (LocalTimeStepping) {
        return db::get<::Tags::Next<::Tags::TimeStepId>>(box);
      } else {
        return db::get<::Tags::TimeStepId>(box);
      }
    }();

    const int tci_decision =
        db::get<evolution::dg::subcell::Tags::TciDecision>(box);
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
        const size_t rdmp_size = rdmp_tci_data.max_variables_values.size() +
                                 rdmp_tci_data.min_variables_values.size();
        const auto& sliced_data_in_direction = all_sliced_data.at(direction);
        // Allocate with subcell data and rdmp data
        DataVector subcell_data_to_send{sliced_data_in_direction.size() +
                                        rdmp_size};
        // Note: Currently we interpolate our solution to our neighbor FD grid
        // even when grid points align but are oriented differently. There's a
        // possible optimization for the rare (almost never?) edge case where
        // two blocks have the same ghost zone coordinates but have different
        // orientations (e.g. RotatedBricks). Since this shouldn't ever happen
        // outside of tests, we currently don't bother with it. If we wanted to,
        // here's the code:
        //
        // if (not orientation.is_aligned()) {
        //   std::array<size_t, Dim> slice_extents{};
        //   for (size_t d = 0; d < Dim; ++d) {
        //     gsl::at(slice_extents, d) = subcell_mesh.extents(d);
        //   }
        //   gsl::at(slice_extents, direction.dimension()) = ghost_zone_size;
        //   // Need a view so we only get the subcell data and not the rdmp
        //   // data
        //   DataVector subcell_data_to_send_view{
        //       subcell_data_to_send.data(),
        //       subcell_data_to_send.size() - rdmp_size};
        //   orient_variables(make_not_null(&subcell_data_to_send_view),
        //                  sliced_data_in_direction, Index<Dim>{slice_extents},
        //                  orientation);
        // } else { std::copy(...); }
        //
        // Copy over data since it's already oriented from interpolation
        std::copy(sliced_data_in_direction.begin(),
                  sliced_data_in_direction.end(), subcell_data_to_send.begin());
        // Copy rdmp data to end of subcell_data_to_send
        std::copy(
            rdmp_tci_data.max_variables_values.cbegin(),
            rdmp_tci_data.max_variables_values.cend(),
            std::prev(subcell_data_to_send.end(), static_cast<int>(rdmp_size)));
        std::copy(rdmp_tci_data.min_variables_values.cbegin(),
                  rdmp_tci_data.min_variables_values.cend(),
                  std::prev(subcell_data_to_send.end(),
                            static_cast<int>(
                                rdmp_tci_data.min_variables_values.size())));

        evolution::dg::BoundaryData<Dim> data{
            subcell_mesh,
            dg_mesh.slice_away(direction.dimension()),
            std::move(subcell_data_to_send),
            std::nullopt,
            next_time_step_id,
            tci_decision};

        Parallel::receive_data<
            evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
            receiver_proxy[neighbor], time_step_id,
            std::pair{DirectionalId<Dim>{direction_from_neighbor, element.id()},
                      std::move(data)});
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Receive the subcell data from our neighbor, and accumulate the data
 * from the relaxed discrete maximum principle troubled-cell indicator.
 *
 * Note:
 * - Since we only care about the min/max over all neighbors and ourself at the
 *   past time, we accumulate all data immediately into the `RdmpTciData`.
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
 *   - `subcell::Tags::GhostDataForReconstruction<Dim>`
 *   - `subcell::Tags::DataForRdmpTci`
 *   - `evolution::dg::Tags::MortarData`
 *   - `evolution::dg::Tags::MortarNextTemporalId`
 */
template <size_t Dim>
struct ReceiveDataForReconstruction {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const auto number_of_expected_messages = element.neighbors().size();
    if (UNLIKELY(number_of_expected_messages == 0)) {
      // We have no neighbors, so just continue without doing any work
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    using ::operator<<;
    using Key = DirectionalId<Dim>;
    const auto& current_time_step_id = db::get<::Tags::TimeStepId>(box);
    std::map<TimeStepId,
             DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>>>& inbox =
        tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
            Metavariables::volume_dim>>(inboxes);
    const auto& received = inbox.find(current_time_step_id);
    // Check we have at least some data from correct time, and then check that
    // we have received all data
    if (received == inbox.end() or
        received->second.size() != number_of_expected_messages) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    // Now that we have received all the data, copy it over as needed.
    DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>> received_data =
        std::move(inbox[current_time_step_id]);
    inbox.erase(current_time_step_id);

    const Mesh<Dim>& subcell_mesh = db::get<Tags::Mesh<Dim>>(box);

    db::mutate<Tags::GhostDataForReconstruction<Dim>, Tags::DataForRdmpTci,
               evolution::dg::Tags::MortarData<Dim>,
               evolution::dg::Tags::MortarNextTemporalId<Dim>,
               domain::Tags::NeighborMesh<Dim>,
               evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
        [&current_time_step_id, &element,
         ghost_zone_size =
             db::get<evolution::dg::subcell::Tags::Reconstructor>(box)
                 .ghost_zone_size(),
         &received_data, &subcell_mesh](
            const gsl::not_null<DirectionalIdMap<Dim, GhostData>*>
                ghost_data_ptr,
            const gsl::not_null<RdmpTciData*> rdmp_tci_data_ptr,
            const gsl::not_null<std::unordered_map<
                Key, evolution::dg::MortarData<Dim>, boost::hash<Key>>*>
                mortar_data,
            const gsl::not_null<
                std::unordered_map<Key, TimeStepId, boost::hash<Key>>*>
                mortar_next_time_step_id,
            const gsl::not_null<DirectionalIdMap<Dim, Mesh<Dim>>*>
                neighbor_mesh,
            const auto neighbor_tci_decisions,
            const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
                neighbor_dg_to_fd_interpolants) {
          // Remove neighbor meshes for neighbors that don't exist anymore
          domain::remove_nonexistent_neighbors(neighbor_mesh, element);

          // Get the next time step id, and also the fluxes data if the neighbor
          // is doing DG.
          for (auto& received_mortar_data : received_data) {
            const auto& mortar_id = received_mortar_data.first;
            try {
              mortar_next_time_step_id->at(mortar_id) =
                  received_mortar_data.second.validity_range;
            } catch (std::exception& e) {
              ERROR("Failed retrieving the MortarId: ("
                    << mortar_id.direction() << ',' << mortar_id.id()
                    << ") from the mortar_next_time_step_id. Got exception: "
                    << e.what());
            }
            if (received_mortar_data.second.boundary_correction_data
                    .has_value()) {
              mortar_data->at(mortar_id).insert_neighbor_mortar_data(
                  current_time_step_id,
                  received_mortar_data.second.interface_mesh,
                  std::move(
                      *received_mortar_data.second.boundary_correction_data));
            }
            // Set new neighbor mesh
            neighbor_mesh->insert_or_assign(
                mortar_id,
                received_mortar_data.second.volume_mesh_ghost_cell_data);
          }

          ASSERT(ghost_data_ptr->empty(),
                 "Should have no elements in the neighbor data when "
                 "receiving neighbor data");
          const size_t number_of_rdmp_vars =
              rdmp_tci_data_ptr->max_variables_values.size();
          ASSERT(rdmp_tci_data_ptr->min_variables_values.size() ==
                     number_of_rdmp_vars,
                 "The number of RDMP variables for which we have a maximum "
                 "and minimum should be the same, but we have "
                     << number_of_rdmp_vars << " for the max and "
                     << rdmp_tci_data_ptr->min_variables_values.size()
                     << " for the min.");

          for (const auto& [direction, neighbors_in_direction] :
               element.neighbors()) {
            for (const auto& neighbor : neighbors_in_direction) {
              DirectionalId<Dim> directional_element_id{direction, neighbor};
              ASSERT(ghost_data_ptr->count(directional_element_id) == 0,
                     "Found neighbor already inserted in direction "
                         << direction << " with ElementId " << neighbor);
              ASSERT(received_data[directional_element_id]
                         .ghost_cell_data.has_value(),
                     "Received subcell data message that does not contain any "
                     "actual subcell data for reconstruction.");
              // Collect the max/min of u(t^n) for the RDMP as we receive data.
              // This reduces the memory footprint.

              evolution::dg::subcell::insert_neighbor_rdmp_and_volume_data(
                  rdmp_tci_data_ptr, ghost_data_ptr,
                  *received_data[directional_element_id].ghost_cell_data,
                  number_of_rdmp_vars, directional_element_id,
                  neighbor_mesh->at(directional_element_id), element,
                  subcell_mesh, ghost_zone_size,
                  neighbor_dg_to_fd_interpolants);
              ASSERT(neighbor_tci_decisions->contains(directional_element_id),
                     "The NeighorTciDecisions should contain the neighbor ("
                         << directional_element_id.direction() << ", "
                         << directional_element_id.id() << ") but doesn't");
              neighbor_tci_decisions->at(directional_element_id) =
                  received_data[directional_element_id].tci_status;
            }
          }
        },
        make_not_null(&box),
        db::get<
            evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<Dim>>(
            box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::subcell::Actions
