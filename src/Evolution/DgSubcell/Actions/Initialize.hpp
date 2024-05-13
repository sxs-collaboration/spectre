// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/InitialTciData.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/StepsSinceTciCall.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciCallsSinceRollback.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Initialize the subcell grid, including the size of the evolved
 * `Variables` and, if present, primitive `Variables`.
 *
 * By default sets the element to `subcell::ActiveGrid::Subcell` unless it
 * is not allowed to use subcell either because it is at an external boundary
 * or because it or one of its neighbors has been marked as DG-only.
 *
 * GlobalCache:
 * - Uses:
 *   - `subcell::Tags::SubcellOptions`
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Mesh<Dim>`
 *   - `domain::Tags::Element<Dim>`
 *   - `System::variables_tag`
 * - Adds:
 *   - `subcell::Tags::Mesh<Dim>`
 *   - `subcell::Tags::ActiveGrid`
 *   - `subcell::Tags::DidRollback`
 *   - `subcell::Tags::TciGridHistory`
 *   - `subcell::Tags::TciCallsSinceRollback`
 *   - `subcell::Tags::GhostDataForReconstruction<Dim>`
 *   - `subcell::Tags::TciDecision`
 *   - `subcell::Tags::DataForRdmpTci`
 *   - `subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>`
 *   - `subcell::fd::Tags::DetInverseJacobianLogicalToGrid`
 *   - `subcell::Tags::LogicalCoordinates<Dim>`
 *   - `subcell::Tags::ReconstructionOrder<Dim>` (set as `std::nullopt`)
 *   - `subcell::Tags::Coordinates<Dim, Frame::Grid>` (as compute tag)
 *   - `subcell::Tags::Coordinates<Dim, Frame::Inertial>` (as compute tag)
 * - Removes: nothing
 * - Modifies:
 *   - `System::variables_tag` and `System::primitive_variables_tag` if the cell
 *     is troubled
 *   - `Tags::dt<System::variables_tag>` if the cell is troubled
 */
template <size_t Dim, typename System, bool UseNumericInitialData>
struct SetSubcellGrid {
  using const_global_cache_tags = tmpl::list<Tags::SubcellOptions<Dim>>;

  using simple_tags = tmpl::list<
      Tags::ActiveGrid, Tags::DidRollback, Tags::TciGridHistory,
      Tags::TciCallsSinceRollback, Tags::StepsSinceTciCall,
      Tags::GhostDataForReconstruction<Dim>, Tags::TciDecision,
      Tags::NeighborTciDecisions<Dim>, Tags::DataForRdmpTci,
      subcell::Tags::CellCenteredFlux<typename System::flux_variables, Dim>,
      subcell::Tags::ReconstructionOrder<Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromDgToNeighborFd<Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<Dim>>;
  using compute_tags =
      tmpl::list<Tags::MeshCompute<Dim>, Tags::LogicalCoordinatesCompute<Dim>,
                 ::domain::Tags::MappedCoordinates<
                     ::domain::Tags::ElementMap<Dim, Frame::Grid>,
                     subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
                     subcell::Tags::Coordinates>,
                 Tags::InertialCoordinatesCompute<
                     ::domain::CoordinateMaps::Tags::CoordinateMap<
                         Dim, Frame::Grid, Frame::Inertial>>,
                 fd::Tags::InverseJacobianLogicalToGridCompute<
                     ::domain::Tags::ElementMap<Dim, Frame::Grid>, Dim>,
                 fd::Tags::DetInverseJacobianLogicalToGridCompute<Dim>,
                 fd::Tags::InverseJacobianLogicalToInertialCompute<
                     ::domain::CoordinateMaps::Tags::CoordinateMap<
                         Dim, Frame::Grid, Frame::Inertial>,
                     Dim>,
                 fd::Tags::DetInverseJacobianLogicalToInertialCompute<
                     ::domain::CoordinateMaps::Tags::CoordinateMap<
                         Dim, Frame::Grid, Frame::Inertial>,
                     Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      [[maybe_unused]] const tuples::TaggedTuple<InboxTags...>& inboxes,
      [[maybe_unused]] const Parallel::GlobalCache<Metavariables>& cache,
      [[maybe_unused]] const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const SubcellOptions& subcell_options =
        db::get<Tags::SubcellOptions<Dim>>(box);
    const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
    const Mesh<Dim>& subcell_mesh = db::get<subcell::Tags::Mesh<Dim>>(box);
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);

    for (size_t d = 0; d < Dim; ++d) {
      if (subcell_options.persson_num_highest_modes() >= dg_mesh.extents(d)) {
        ERROR("Number of the highest modes to be monitored by the Persson TCI ("
              << subcell_options.persson_num_highest_modes()
              << ") must be smaller than the extent of the DG mesh ("
              << dg_mesh.extents(d) << ").");
      }
    }

    // Loop over block neighbors and if neighbor id is inside of
    // subcell_options.only_dg_block_ids(), then bordering DG-only block
    const bool bordering_dg_block = alg::any_of(
        element.neighbors(),
        [&subcell_options](const auto& direction_and_neighbor) {
          const size_t first_block_id =
              direction_and_neighbor.second.ids().begin()->block_id();
          return std::binary_search(subcell_options.only_dg_block_ids().begin(),
                                    subcell_options.only_dg_block_ids().end(),
                                    first_block_id);
        });

    const bool subcell_allowed_in_element =
        not std::binary_search(subcell_options.only_dg_block_ids().begin(),
                               subcell_options.only_dg_block_ids().end(),
                               element.id().block_id()) and
        not bordering_dg_block;
    const bool cell_is_not_on_external_boundary =
        db::get<::domain::Tags::Element<Dim>>(box)
            .external_boundaries()
            .empty();

    constexpr bool subcell_enabled_at_external_boundary =
        Metavariables::SubcellOptions::subcell_enabled_at_external_boundary;

    db::mutate<Tags::NeighborTciDecisions<Dim>>(
        [&element](const auto neighbor_decisions_ptr) {
          neighbor_decisions_ptr->clear();
          for (const auto& [direction, neighbors_in_direction] :
               element.neighbors()) {
            for (const auto& neighbor : neighbors_in_direction.ids()) {
              neighbor_decisions_ptr->insert(
                  std::pair{DirectionalId<Dim>{direction, neighbor}, 0});
            }
          }
        },
        make_not_null(&box));

    db::mutate_apply<
        tmpl::list<Tags::ActiveGrid, Tags::DidRollback,
                   typename System::variables_tag, subcell::Tags::TciDecision,
                   subcell::Tags::TciCallsSinceRollback,
                   subcell::Tags::StepsSinceTciCall>,
        tmpl::list<>>(
        [&cell_is_not_on_external_boundary, &dg_mesh,
         subcell_allowed_in_element, &subcell_mesh](
            const gsl::not_null<ActiveGrid*> active_grid_ptr,
            const gsl::not_null<bool*> did_rollback_ptr,
            const auto active_vars_ptr,
            const gsl::not_null<int*> tci_decision_ptr,
            const gsl::not_null<size_t*> tci_calls_since_rollback_ptr,
            const gsl::not_null<size_t*> steps_since_tci_call_ptr) {
          // We don't consider setting the initial grid to subcell as rolling
          // back. Since no time step is undone, we just continue on the
          // subcells as a normal solve.
          *did_rollback_ptr = false;

          if ((cell_is_not_on_external_boundary or
               subcell_enabled_at_external_boundary) and
              subcell_allowed_in_element) {
            *active_grid_ptr = ActiveGrid::Subcell;
            active_vars_ptr->initialize(subcell_mesh.number_of_grid_points());
          } else {
            *active_grid_ptr = ActiveGrid::Dg;
            active_vars_ptr->initialize(dg_mesh.number_of_grid_points());
          }

          *tci_decision_ptr = 0;
          *tci_calls_since_rollback_ptr = 0;
          *steps_since_tci_call_ptr = 0;
        },
        make_not_null(&box));
    if constexpr (System::has_primitive_and_conservative_vars) {
      db::mutate<typename System::primitive_variables_tag>(
          [&dg_mesh, &subcell_mesh](const auto prim_vars_ptr,
                                    const auto active_grid) {
            if (active_grid == ActiveGrid::Dg) {
              prim_vars_ptr->initialize(dg_mesh.number_of_grid_points());
            } else {
              prim_vars_ptr->initialize(subcell_mesh.number_of_grid_points());
            }
          },
          make_not_null(&box), db::get<Tags::ActiveGrid>(box));
    }
    if constexpr (not UseNumericInitialData) {
      if (db::get<Tags::ActiveGrid>(box) ==
          evolution::dg::subcell::ActiveGrid::Dg) {
        evolution::Initialization::Actions::SetVariables<
            ::domain::Tags::Coordinates<Dim, Frame::ElementLogical>>::
            apply(box, inboxes, cache, array_index, ActionList{},
                  std::add_pointer_t<ParallelComponent>{nullptr});
      } else {
        evolution::Initialization::Actions::
            SetVariables<Tags::Coordinates<Dim, Frame::ElementLogical>>::apply(
                box, inboxes, cache, array_index, ActionList{},
                std::add_pointer_t<ParallelComponent>{nullptr});
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Sets the RDMP data from the initial data and sends it to neighboring
 * elements.
 *
 * GlobalCache:
 * - Uses:
 *   - `ParallelComponent` proxy
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Element<Dim>`
 *   - `subcell::Tags::DataForRdmpTci`
 *   - `subcell::Tags::InitialTciData`
 *   - whatever `SetInitialRdmpData` uses
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - whatever `SetInitialRdmpData` mutates
 */
template <size_t Dim, typename SetInitialRdmpData>
struct SetAndCommunicateInitialRdmpData {
  using inbox_tags =
      tmpl::list<evolution::dg::subcell::Tags::InitialTciData<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Get the RDMP data on this element and then initialize it.
    db::mutate_apply<SetInitialRdmpData>(make_not_null(&box));

    // Send RDMP data to neighbors
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& rdmp_data =
        db::get<evolution::dg::subcell::Tags::DataForRdmpTci>(box);
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor : neighbors) {
        evolution::dg::subcell::InitialTciData data{{}, rdmp_data};
        // We use temporal ID 0 for sending RDMP data
        const int temporal_id = 0;
        Parallel::receive_data<
            evolution::dg::subcell::Tags::InitialTciData<Dim>>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                DirectionalId<Dim>{direction_from_neighbor, element.id()},
                std::move(data)));
      }
    }

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Apply the TCI on the FD grid to the initial data and send the TCI
 * decision to neighboring elements.
 *
 * GlobalCache:
 * - Uses:
 *   - `ParallelComponent` proxy
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Element<Dim>`
 *   - `subcell::Tags::DataForRdmpTci`
 *   - `subcell::Tags::InitialTciData`
 *   - `subcell::Tags::SubcellOptions`
 *   - `subcell::Tags::ActiveGrid`
 *   - whatever `TciOnFdGridMutator` uses
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::Tags::DataForRdmpTci`
 *   - `subcell::Tags::TciDecision`
 */
template <size_t Dim, typename System, typename TciOnFdGridMutator>
struct ComputeAndSendTciOnInitialGrid {
  using inbox_tags =
      tmpl::list<evolution::dg::subcell::Tags::InitialTciData<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);

    // Check if we have received all RDMP data.
    if (LIKELY(element.number_of_neighbors() != 0)) {
      auto& inbox =
          tuples::get<evolution::dg::subcell::Tags::InitialTciData<Dim>>(
              inboxes);
      const auto& received = inbox.find(0);
      if (received == inbox.end() or
          received->second.size() != element.number_of_neighbors()) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }

      db::mutate<evolution::dg::subcell::Tags::DataForRdmpTci>(
          [&element, &received](const auto rdmp_tci_data_ptr) {
            (void)element;
            const size_t number_of_rdmp_vars =
                rdmp_tci_data_ptr->max_variables_values.size();
            ASSERT(rdmp_tci_data_ptr->max_variables_values.size() ==
                       number_of_rdmp_vars,
                   "The number of local max vars is "
                       << number_of_rdmp_vars
                       << " while the number of local min vars is "
                       << rdmp_tci_data_ptr->max_variables_values.size()
                       << " the local element ID is " << element.id());
            for (const auto& [direction_and_neighbor_element_id,
                              neighbor_initial_tci_data] : received->second) {
              ASSERT(neighbor_initial_tci_data.initial_rdmp_data.has_value(),
                     "Neighbor in direction "
                         << direction_and_neighbor_element_id.direction()
                         << " with element ID "
                         << direction_and_neighbor_element_id.id() << " of "
                         << element.id()
                         << " didn't send initial TCI data correctly");
              ASSERT(
                  neighbor_initial_tci_data.initial_rdmp_data.value()
                          .max_variables_values.size() == number_of_rdmp_vars,
                  "The number of local RDMP vars is "
                      << number_of_rdmp_vars
                      << " while the number of remote max vars is "
                      << neighbor_initial_tci_data.initial_rdmp_data.value()
                             .max_variables_values.size()
                      << " the local element ID is " << element.id()
                      << " and the remote id is "
                      << direction_and_neighbor_element_id.id());
              ASSERT(
                  neighbor_initial_tci_data.initial_rdmp_data.value()
                          .min_variables_values.size() == number_of_rdmp_vars,
                  "The number of local RDMP vars is "
                      << number_of_rdmp_vars
                      << " while the number of remote min vars is "
                      << neighbor_initial_tci_data.initial_rdmp_data.value()
                             .min_variables_values.size()
                      << " the local element ID is " << element.id()
                      << " and the remote id is "
                      << direction_and_neighbor_element_id.id());
              for (size_t var_index = 0; var_index < number_of_rdmp_vars;
                   ++var_index) {
                rdmp_tci_data_ptr->max_variables_values[var_index] =
                    std::max(rdmp_tci_data_ptr->max_variables_values[var_index],
                             neighbor_initial_tci_data.initial_rdmp_data.value()
                                 .max_variables_values[var_index]);
                rdmp_tci_data_ptr->min_variables_values[var_index] =
                    std::min(rdmp_tci_data_ptr->min_variables_values[var_index],
                             neighbor_initial_tci_data.initial_rdmp_data.value()
                                 .min_variables_values[var_index]);
              }
            }
          },
          make_not_null(&box));
      inbox.erase(received);
    }

    const auto send_tci_decision = [&cache, &element](const int tci_decision) {
      if (UNLIKELY(element.number_of_neighbors() == 0)) {
        return;
      }
      auto& receiver_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      for (const auto& [direction, neighbors] : element.neighbors()) {
        const auto& orientation = neighbors.orientation();
        const auto direction_from_neighbor = orientation(direction.opposite());
        for (const auto& neighbor : neighbors) {
          evolution::dg::subcell::InitialTciData data{tci_decision, {}};
          // We use temporal ID 1 for ending the TCI decision.
          const int temporal_id = 1;
          Parallel::receive_data<
              evolution::dg::subcell::Tags::InitialTciData<Dim>>(
              receiver_proxy[neighbor], temporal_id,
              std::make_pair(
                  DirectionalId<Dim>{direction_from_neighbor, element.id()},
                  std::move(data)));
        }
      }
    };

    const SubcellOptions& subcell_options =
        db::get<Tags::SubcellOptions<Dim>>(box);

    if (subcell_options.always_use_subcells() or
        get<Tags::ActiveGrid>(box) == ActiveGrid::Dg) {
      db::mutate<Tags::TciDecision>(
          [](const gsl::not_null<int*> tci_decision_ptr) {
            *tci_decision_ptr = 0;
          },
          make_not_null(&box));
      send_tci_decision(0);
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    // Now run the TCI to see if we could switch back to DG.
    const std::tuple<int, evolution::dg::subcell::RdmpTciData> tci_result =
        db::mutate_apply<TciOnFdGridMutator>(
            make_not_null(&box), subcell_options.persson_exponent() + 1.0,
            false);

    db::mutate<Tags::TciDecision>(
        [&tci_result](const gsl::not_null<int*> tci_decision_ptr) {
          *tci_decision_ptr = std::get<0>(tci_result);
        },
        make_not_null(&box));
    send_tci_decision(std::get<0>(tci_result));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Using the local and neighboring TCI decisions, switches the element to
 * DG if the DG solution was determined to be admissible.
 *
 * GlobalCache:
 * - Uses:
 *   - `ParallelComponent` proxy
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Element<Dim>`
 *   - `subcell::Tags::DataForRdmpTci`
 *   - `subcell::Tags::InitialTciData`
 *   - `subcell::Tags::SubcellOptions`
 *   - `subcell::Tags::ActiveGrid`
 *   - whatever `TciOnFdGridMutator` uses
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::Tags::NeighborTciDecisions`
 *   - `System::variables_tag`
 *   - `Tags::HistoryEvolvedVariables<System::variables_tag>`
 *   - `subcell::Tags::GhostDataForReconstruction`
 *   - `subcell::Tags::TciGridHistory`
 *   - `subcell::Tags::CellCenteredFlux`
 */
template <size_t Dim, typename System>
struct SetInitialGridFromTciData {
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    if (LIKELY(element.number_of_neighbors() != 0)) {
      auto& inbox =
          tuples::get<evolution::dg::subcell::Tags::InitialTciData<Dim>>(
              inboxes);
      const auto& received = inbox.find(1);
      // Check if we have received all TCI decisions.
      if (received == inbox.end() or
          received->second.size() != element.number_of_neighbors()) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }

      db::mutate<evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
          [&element, &received](const auto neighbor_tci_decisions_ptr) {
            for (const auto& [directional_element_id,
                              neighbor_initial_tci_data] : received->second) {
              (void)element;
              ASSERT(neighbor_initial_tci_data.tci_status.has_value(),
                     "Neighbor in direction "
                         << directional_element_id.direction()
                         << " with element ID " << directional_element_id.id()
                         << " of " << element.id()
                         << " didn't send initial TCI decision correctly");
              neighbor_tci_decisions_ptr->at(directional_element_id) =
                  neighbor_initial_tci_data.tci_status.value();
            }
          },
          make_not_null(&box));
      inbox.erase(received);
    }

    if (get<Tags::ActiveGrid>(box) == ActiveGrid::Dg) {
      // In this case we are allowed to only do DG in this element. No need to
      // even do any checks.
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const SubcellOptions& subcell_options =
        db::get<Tags::SubcellOptions<Dim>>(box);

    bool cell_is_troubled =
        subcell_options.always_use_subcells() or
        (subcell_options.use_halo() and [&box]() -> bool {
          for (const auto& [_, neighbor_decision] :
               db::get<evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
                   box)) {
            if (neighbor_decision != 0) {
              return true;
            }
          }
          return false;
        }()) or
        (db::get<Tags::TciDecision>(box) != 0);

    if (not cell_is_troubled) {
      using variables_tag = typename System::variables_tag;
      using flux_variables = typename System::flux_variables;

      const Mesh<Dim>& dg_mesh = db::get<::domain::Tags::Mesh<Dim>>(box);
      const Mesh<Dim>& subcell_mesh = db::get<subcell::Tags::Mesh<Dim>>(box);
      db::mutate<
          variables_tag, ::Tags::HistoryEvolvedVariables<variables_tag>,
          Tags::ActiveGrid, subcell::Tags::GhostDataForReconstruction<Dim>,
          evolution::dg::subcell::Tags::TciGridHistory,
          evolution::dg::subcell::Tags::CellCenteredFlux<flux_variables, Dim>>(
          [&dg_mesh, &subcell_mesh, &subcell_options](
              const auto active_vars_ptr, const auto active_history_ptr,
              const gsl::not_null<ActiveGrid*> active_grid_ptr,
              const auto subcell_ghost_data_ptr,
              const gsl::not_null<
                  std::deque<evolution::dg::subcell::ActiveGrid>*>
                  tci_grid_history_ptr,
              const auto subcell_cell_centered_fluxes) {
            // Note: strictly speaking, to be conservative this should
            // reconstruct uJ instead of u.
            *active_vars_ptr = fd::reconstruct(
                *active_vars_ptr, dg_mesh, subcell_mesh.extents(),
                subcell_options.reconstruction_method());

            // Reconstruct the DG solution for each time in the time stepper
            // history
            active_history_ptr->map_entries(
                [&dg_mesh, &subcell_mesh, &subcell_options](const auto entry) {
                  *entry =
                      fd::reconstruct(*entry, dg_mesh, subcell_mesh.extents(),
                                      subcell_options.reconstruction_method());
                });
            *active_grid_ptr = ActiveGrid::Dg;

            // Clear the neighbor data needed for subcell reconstruction since
            // we have now completed the time step.
            subcell_ghost_data_ptr->clear();

            // Clear the TCI grid history since we don't need to use it when on
            // the DG grid.
            tci_grid_history_ptr->clear();

            // Clear the allocation for the cell-centered fluxes.
            *subcell_cell_centered_fluxes = std::nullopt;
          },
          make_not_null(&box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::dg::subcell::Actions
