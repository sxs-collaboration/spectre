// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/AsAccess.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/TrimMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/SelfStart.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
struct Time;
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags

namespace evolution::dg::subcell {
// We use a forward declaration instead of including a header file to avoid
// coupling to the DG-subcell libraries for executables that don't use subcell.
template <size_t VolumeDim, typename DgComputeSubcellNeighborPackagedData>
void neighbor_reconstructed_face_solution(
    gsl::not_null<db::Access*> box,
    gsl::not_null<std::pair<
        TimeStepId,
        DirectionalIdMap<VolumeDim, evolution::dg::BoundaryData<VolumeDim>>>*>
        received_temporal_id_and_data);
template <size_t Dim>
void neighbor_tci_decision(
    gsl::not_null<db::Access*> box,
    const std::pair<TimeStepId,
                    DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>>>&
        received_temporal_id_and_data);
}  // namespace evolution::dg::subcell
/// \endcond

namespace evolution::dg {
namespace detail {
template <typename BoundaryCorrectionClass>
struct get_dg_boundary_terms {
  using type = typename BoundaryCorrectionClass::dg_boundary_terms_volume_tags;
};

template <typename Tag, typename Type = db::const_item_type<Tag, tmpl::list<>>>
struct TemporaryReference {
  using tag = Tag;
  using type = const Type&;
};

template <size_t Dim>
void retrieve_boundary_data_spsc(
    const gsl::not_null<std::map<
        TimeStepId, DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>>>*>
        boundary_data_ptr,
    const gsl::not_null<evolution::dg::AtomicInboxBoundaryData<Dim>*> inbox_ptr,
    const Element<Dim>& element) {
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const ElementId<Dim>& neighbor_element_id : neighbors) {
      const size_t neighbor_index =
          inbox_ptr->index(DirectionalId{direction, neighbor_element_id});
      auto& spsc_in_direction =
          gsl::at(inbox_ptr->boundary_data_in_directions, neighbor_index);
      auto* data_in_direction = spsc_in_direction.front();
      while (data_in_direction != nullptr) {
        const auto& time_step_id = get<0>(*data_in_direction);
        auto& data = get<1>(*data_in_direction);
        auto& directional_element_id = get<2>(*data_in_direction);
        auto& current_inbox = (*boundary_data_ptr)[time_step_id];
        if (auto it = current_inbox.find(directional_element_id);
            it != current_inbox.end()) {
          auto& [volume_mesh, volume_mesh_of_ghost_cell_data, face_mesh,
                 ghost_cell_data, boundary_data, boundary_data_validity_range,
                 boundary_tci_status, boundary_integration_order] = data;
          (void)ghost_cell_data;
          auto& [current_volume_mesh, current_volume_mesh_of_ghost_cell_data,
                 current_face_mesh, current_ghost_cell_data,
                 current_boundary_data, current_boundary_data_validity_range,
                 current_tci_status, current_integration_order] = it->second;
          // Need to use when optimizing subcell
          (void)current_volume_mesh_of_ghost_cell_data;
          // We have already received some data at this time. Receiving
          // data twice at the same time should only occur when
          // receiving fluxes after having previously received ghost
          // cells. We sanity check that the data we already have is the
          // ghost cells and that we have not yet received flux data.
          //
          // This is used if a 2-send implementation is used (which we
          // don't right now!). We generally find that the number of
          // communications is more important than the size of each
          // communication, and so a single communication per time/sub
          // step is preferred.
          ASSERT(current_ghost_cell_data.has_value(),
                 "Have not yet received ghost cells at time step "
                     << time_step_id
                     << " but the inbox entry already exists. This is a bug in "
                        "the ordering of the actions.");
          ASSERT(not current_boundary_data.has_value(),
                 "The fluxes have already been received at time step "
                     << time_step_id
                     << ". They are either being received for a second time, "
                        "there is a bug in the ordering of the actions (though "
                        "a different ASSERT should've caught that), or the "
                        "incorrect temporal ID is being sent.");

          ASSERT(current_face_mesh == face_mesh,
                 "The mesh being received for the fluxes is different than the "
                 "mesh received for the ghost cells. Mesh for fluxes: "
                     << face_mesh << " mesh for ghost cells "
                     << current_face_mesh);
          ASSERT(current_volume_mesh_of_ghost_cell_data ==
                     volume_mesh_of_ghost_cell_data,
                 "The mesh being received for the ghost cell data is different "
                 "than the mesh received previously. Mesh for received when we "
                 "got fluxes: "
                     << volume_mesh_of_ghost_cell_data
                     << " mesh received when we got ghost cells "
                     << current_volume_mesh_of_ghost_cell_data);

          // We always move here since we take ownership of the data and
          // moves implicitly decay to copies
          current_boundary_data = std::move(boundary_data);
          current_boundary_data_validity_range = boundary_data_validity_range;
          current_tci_status = boundary_tci_status;
          current_integration_order = boundary_integration_order;
        } else {
          // We have not received ghost cells or fluxes at this time.
          if (not current_inbox
                      .emplace(std::move(directional_element_id),
                               std::move(data))
                      .second) {
            ERROR("Failed to insert data to receive at instance '"
                  << time_step_id
                  << "' with tag 'BoundaryCorrectionAndGhostCellsInbox'.\n");
          }
        }

        spsc_in_direction.pop();
        data_in_direction = spsc_in_direction.front();
      }  // while data_in_direction != nullptr
    }    // for neighbor_element_id : neighbors
  }      // for element.neighbors()
}
}  // namespace detail

/// Receive boundary data for global time-stepping.  Returns true if
/// all necessary data has been received.
template <bool UseNodegroupDgElements, typename Metavariables,
          typename DbTagsList, typename... InboxTags>
bool receive_boundary_data_global_time_stepping(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes) {
  constexpr size_t volume_dim = Metavariables::system::volume_dim;

  const TimeStepId& temporal_id = get<::Tags::TimeStepId>(*box);
  using Key = DirectionalId<volume_dim>;
  using InboxMap = std::map<
      TimeStepId,
      DirectionalIdMap<volume_dim, evolution::dg::BoundaryData<volume_dim>>>;
  using InboxMapValueType =
      std::pair<typename InboxMap::key_type, typename InboxMap::mapped_type>;
  using NodeType = typename InboxMap::node_type;
  auto get_temporal_id_and_data_node =
      [&temporal_id](const gsl::not_null<InboxMap*> map_ptr,
                     const Element<volume_dim>& element,
                     const auto&&...) -> NodeType {
    const auto received_temporal_id_and_data = map_ptr->find(temporal_id);
    if (received_temporal_id_and_data == map_ptr->end()) {
      return NodeType{};
    }
    const auto& received_neighbor_data = received_temporal_id_and_data->second;
    for (const auto& [direction, neighbors] : element.neighbors()) {
      for (const auto& neighbor : neighbors) {
        const auto neighbor_received =
            received_neighbor_data.find(Key{direction, neighbor});
        if (neighbor_received == received_neighbor_data.end()) {
          return NodeType{};
        }
      }
    }
    return map_ptr->extract(received_temporal_id_and_data);
  };

  InboxMapValueType received_temporal_id_and_data{};
  if constexpr (std::is_same_v<
                    evolution::dg::AtomicInboxBoundaryData<volume_dim>,
                    typename evolution::dg::Tags::
                        BoundaryCorrectionAndGhostCellsInbox<
                            volume_dim, UseNodegroupDgElements>::type>) {
    bool have_all_data = false;
    received_temporal_id_and_data =
        db::mutate<evolution::dg::Tags::BoundaryData<volume_dim>>(
            [&get_temporal_id_and_data_node](
                const auto boundary_data_ptr,
                const gsl::not_null<bool*> local_have_all_data,
                const auto inbox_ptr, const Element<volume_dim>& element)
                -> std::pair<typename NodeType::key_type,
                             typename NodeType::mapped_type> {
              if (inbox_ptr->message_count.load(std::memory_order_relaxed) <
                  element.number_of_neighbors()) {
                return {};
              }
              detail::retrieve_boundary_data_spsc(boundary_data_ptr, inbox_ptr,
                                                  element);

              NodeType node =
                  get_temporal_id_and_data_node(boundary_data_ptr, element);
              if (node.empty()) {
                return {};
              }
              if (UNLIKELY(node.mapped().size() !=
                           element.number_of_neighbors())) {
                ERROR("Incorrect number of element neighbors");
              }
              *local_have_all_data = true;
              // We only decrease the counter if we are done with the current
              // time and we only decrease it by the number of neighbors at the
              // current time.
              inbox_ptr->message_count.fetch_sub(element.number_of_neighbors(),
                                                 std::memory_order_acq_rel);

              return std::pair{std::move(node.key()), std::move(node.mapped())};
            },
            box, make_not_null(&have_all_data),
            make_not_null(
                &tuples::get<
                    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                        volume_dim, UseNodegroupDgElements>>(*inboxes)),
            db::get<domain::Tags::Element<volume_dim>>(*box));
    if (not have_all_data) {
      return false;
    }
  } else {
    // Scope to make sure the `node` can't be used later.
    NodeType node = get_temporal_id_and_data_node(
        make_not_null(&tuples::get<
                      evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                          volume_dim, UseNodegroupDgElements>>(*inboxes)),
        db::get<domain::Tags::Element<volume_dim>>(*box));
    if (node.empty()) {
      return false;
    }
    received_temporal_id_and_data.first = std::move(node.key());
    received_temporal_id_and_data.second = std::move(node.mapped());
  }

  // Move inbox contents into the DataBox
  if constexpr (using_subcell_v<Metavariables>) {
    evolution::dg::subcell::neighbor_reconstructed_face_solution<
        volume_dim, typename Metavariables::SubcellOptions::
                        DgComputeSubcellNeighborPackagedData>(
        &db::as_access(*box), make_not_null(&received_temporal_id_and_data));
    evolution::dg::subcell::neighbor_tci_decision<volume_dim>(
        make_not_null(&db::as_access(*box)), received_temporal_id_and_data);
  }

  const auto& mortar_meshes =
      get<evolution::dg::Tags::MortarMesh<volume_dim>>(*box);
  db::mutate<evolution::dg::Tags::MortarData<volume_dim>,
             evolution::dg::Tags::MortarNextTemporalId<volume_dim>,
             domain::Tags::NeighborMesh<volume_dim>>(
      [&received_temporal_id_and_data, &mortar_meshes](
          const gsl::not_null<DirectionalIdMap<
              volume_dim, evolution::dg::MortarDataHolder<volume_dim>>*>
              mortar_data,
          const gsl::not_null<DirectionalIdMap<volume_dim, TimeStepId>*>
              mortar_next_time_step_id,
          const gsl::not_null<DirectionalIdMap<volume_dim, Mesh<volume_dim>>*>
              neighbor_mesh) {
        neighbor_mesh->clear();
        for (auto& received_mortar_data :
             received_temporal_id_and_data.second) {
          const auto& mortar_id = received_mortar_data.first;
          neighbor_mesh->insert_or_assign(
              mortar_id, received_mortar_data.second.volume_mesh);
          mortar_next_time_step_id->at(mortar_id) =
              received_mortar_data.second.validity_range;
          ASSERT(using_subcell_v<Metavariables> or
                     received_mortar_data.second.boundary_correction_data
                         .has_value(),
                 "Must receive number boundary correction data when not using "
                 "DG-subcell. Mortar ID is: ("
                     << mortar_id.direction() << "," << mortar_id.id()
                     << ") and TimeStepId is "
                     << received_temporal_id_and_data.first);
          if (received_mortar_data.second.boundary_correction_data
                  .has_value()) {
            mortar_data->at(mortar_id).neighbor().face_mesh =
                received_mortar_data.second.interface_mesh;
            mortar_data->at(mortar_id).neighbor().mortar_mesh =
                mortar_meshes.at(mortar_id);
            mortar_data->at(mortar_id).neighbor().mortar_data = std::move(
                received_mortar_data.second.boundary_correction_data.value());
          }
        }
      },
      box);
  return true;
}

/// Receive boundary data for local time-stepping.  Returns true if
/// all necessary data has been received.
///
/// Setting \p DenseOutput to true receives data required for output
/// at `::Tags::Time` instead of `::Tags::Next<::Tags::TimeStepId>`.
template <bool UseNodegroupDgElements, typename System, size_t Dim,
          bool DenseOutput, typename DbTagsList, typename... InboxTags>
bool receive_boundary_data_local_time_stepping(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes) {
  using variables_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  const auto needed_time = [&box]() {
    const LtsTimeStepper& time_stepper =
        db::get<::Tags::TimeStepper<LtsTimeStepper>>(*box);
    if constexpr (DenseOutput) {
      const auto& dense_output_time = db::get<::Tags::Time>(*box);
      return [&dense_output_time, &time_stepper](const TimeStepId& id) {
        return time_stepper.neighbor_data_required(dense_output_time, id);
      };
    } else {
      const auto& next_temporal_id =
          db::get<::Tags::Next<::Tags::TimeStepId>>(*box);
      return [&next_temporal_id, &time_stepper](const TimeStepId& id) {
        return time_stepper.neighbor_data_required(next_temporal_id, id);
      };
    }
  }();

  // The boundary history coupling computation (which computes the _lifted_
  // boundary correction) returns a Variables<dt<EvolvedVars>> instead of
  // using the `NormalDotNumericalFlux` prefix tag. This is because the
  // returned quantity is more a `dt` quantity than a
  // `NormalDotNormalDotFlux` since it's been lifted to the volume.
  using InboxMap =
      std::map<TimeStepId,
               DirectionalIdMap<Dim, evolution::dg::BoundaryData<Dim>>>;
  InboxMap* inbox_ptr = nullptr;
  if constexpr (std::is_same_v<evolution::dg::AtomicInboxBoundaryData<Dim>,
                               typename evolution::dg::Tags::
                                   BoundaryCorrectionAndGhostCellsInbox<
                                       Dim, UseNodegroupDgElements>::type>) {
    inbox_ptr = db::mutate<evolution::dg::Tags::BoundaryData<Dim>>(
        [](const auto boundary_data_ptr, const auto local_inbox_ptr,
           const Element<Dim>& element) -> InboxMap* {
          detail::retrieve_boundary_data_spsc(boundary_data_ptr,
                                              local_inbox_ptr, element);

          return boundary_data_ptr.get();
        },
        box,
        make_not_null(&tuples::get<
                      evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                          Dim, UseNodegroupDgElements>>(*inboxes)),
        db::get<domain::Tags::Element<Dim>>(*box));
  } else {
    inbox_ptr =
        &tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
            Dim, UseNodegroupDgElements>>(*inboxes);
  }
  ASSERT(inbox_ptr != nullptr, "The inbox pointer should not be null.");
  InboxMap& inbox = *inbox_ptr;
  const auto& mortar_meshes = get<evolution::dg::Tags::MortarMesh<Dim>>(*box);

  const bool have_all_intermediate_messages = db::mutate<
      evolution::dg::Tags::MortarDataHistory<Dim,
                                             typename dt_variables_tag::type>,
      evolution::dg::Tags::MortarNextTemporalId<Dim>,
      domain::Tags::NeighborMesh<Dim>>(
      [&inbox, &needed_time, &mortar_meshes](
          const gsl::not_null<DirectionalIdMap<
              Dim,
              TimeSteppers::BoundaryHistory<evolution::dg::MortarData<Dim>,
                                            evolution::dg::MortarData<Dim>,
                                            typename dt_variables_tag::type>>*>
              boundary_data_history,
          const gsl::not_null<DirectionalIdMap<Dim, TimeStepId>*>
              mortar_next_time_step_ids,
          const gsl::not_null<DirectionalIdMap<Dim, Mesh<Dim>>*> neighbor_mesh,
          const Element<Dim>& element) {
        // Remove neighbor meshes for neighbors that don't exist anymore
        domain::remove_nonexistent_neighbors(neighbor_mesh, element);

        // Move received boundary data into boundary history.
        for (auto& [mortar_id, mortar_next_time_step_id] :
             *mortar_next_time_step_ids) {
          if (mortar_id.id() == ElementId<Dim>::external_boundary_id()) {
            continue;
          }
          while (needed_time(mortar_next_time_step_id)) {
            const auto time_entry = inbox.find(mortar_next_time_step_id);
            if (time_entry == inbox.end()) {
              return false;
            }
            const auto received_mortar_data =
                time_entry->second.find(mortar_id);
            if (received_mortar_data == time_entry->second.end()) {
              return false;
            }

            MortarData<Dim> neighbor_mortar_data{};
            // Insert:
            // - the current TimeStepId of the neighbor
            // - the current face mesh of the neighbor
            // - the current boundary correction data of the neighbor
            ASSERT(received_mortar_data->second.boundary_correction_data
                       .has_value(),
                   "Did not receive boundary correction data from the "
                   "neighbor\nMortarId: "
                       << mortar_id
                       << "\nTimeStepId: " << mortar_next_time_step_id);
            neighbor_mesh->insert_or_assign(
                mortar_id, received_mortar_data->second.volume_mesh);
            neighbor_mortar_data.face_mesh =
                received_mortar_data->second.interface_mesh;
            neighbor_mortar_data.mortar_mesh = mortar_meshes.at(mortar_id);
            neighbor_mortar_data.mortar_data = std::move(
                received_mortar_data->second.boundary_correction_data.value());
            boundary_data_history->at(mortar_id).remote().insert(
                mortar_next_time_step_id,
                received_mortar_data->second.integration_order,
                std::move(neighbor_mortar_data));
            mortar_next_time_step_id =
                received_mortar_data->second.validity_range;
            time_entry->second.erase(received_mortar_data);
            if (time_entry->second.empty()) {
              inbox.erase(time_entry);
            }
          }
        }
        return true;
      },
      box, db::get<::domain::Tags::Element<Dim>>(*box));

  if (not have_all_intermediate_messages) {
    return false;
  }

  if constexpr (std::is_same_v<evolution::dg::AtomicInboxBoundaryData<Dim>,
                               typename evolution::dg::Tags::
                                   BoundaryCorrectionAndGhostCellsInbox<
                                       Dim, UseNodegroupDgElements>::type>) {
    // We only decrease the counter if we are done with the current time
    // and we only decrease it by the number of neighbors at the current
    // time.
    tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
        Dim, UseNodegroupDgElements>>(*inboxes)
        .message_count.fetch_sub(
            db::get<domain::Tags::Element<Dim>>(*box).number_of_neighbors(),
            std::memory_order_acq_rel);
  }
  return have_all_intermediate_messages;
}

/// Apply corrections from boundary communication.
///
/// If `LocalTimeStepping` is false, updates the derivative of the variables,
/// which should be done before taking a time step.  If
/// `LocalTimeStepping` is true, updates the variables themselves, which should
/// be done after the volume update.
///
/// Setting \p DenseOutput to true receives data required for output
/// at ::Tags::Time instead of performing a full step.  This is only
/// used for local time-stepping.
template <bool LocalTimeStepping, typename System, size_t VolumeDim,
          bool DenseOutput>
struct ApplyBoundaryCorrections {
  static constexpr bool local_time_stepping = LocalTimeStepping;
  static_assert(local_time_stepping or not DenseOutput,
                "GTS does not use ApplyBoundaryCorrections for dense output.");

  using system = System;
  static constexpr size_t volume_dim = VolumeDim;
  using variables_tag = typename system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using DtVariables = typename dt_variables_tag::type;
  using volume_tags_for_dg_boundary_terms =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
          typename system::boundary_correction_base::creatable_classes,
          detail::get_dg_boundary_terms<tmpl::_1>>>>;

  using TimeStepperType =
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>;

  using tag_to_update =
      tmpl::conditional_t<local_time_stepping, variables_tag, dt_variables_tag>;
  using mortar_data_tag = tmpl::conditional_t<
      local_time_stepping,
      evolution::dg::Tags::MortarDataHistory<volume_dim, DtVariables>,
      evolution::dg::Tags::MortarData<volume_dim>>;
  using MortarDataType =
      tmpl::conditional_t<DenseOutput, const typename mortar_data_tag::type,
                          typename mortar_data_tag::type>;

  using return_tags =
      tmpl::conditional_t<DenseOutput, tmpl::list<tag_to_update>,
                          tmpl::list<tag_to_update, mortar_data_tag>>;
  using argument_tags = tmpl::append<
      tmpl::flatten<tmpl::list<
          tmpl::conditional_t<DenseOutput, mortar_data_tag, tmpl::list<>>,
          domain::Tags::Mesh<volume_dim>, Tags::MortarMesh<volume_dim>,
          Tags::MortarSize<volume_dim>, ::dg::Tags::Formulation,
          evolution::dg::Tags::NormalCovectorAndMagnitude<volume_dim>,
          ::Tags::TimeStepper<TimeStepperType>,
          evolution::Tags::BoundaryCorrection<system>,
          tmpl::conditional_t<DenseOutput, ::Tags::Time, ::Tags::TimeStep>,
          tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                              domain::Tags::DetInvJacobian<
                                  Frame::ElementLogical, Frame::Inertial>>>>,
      volume_tags_for_dg_boundary_terms>;

  // full step
  template <typename... VolumeArgs>
  static void apply(
      const gsl::not_null<typename tag_to_update::type*> vars_to_update,
      const gsl::not_null<MortarDataType*> mortar_data,
      const Mesh<volume_dim>& volume_mesh,
      const typename Tags::MortarMesh<volume_dim>::type& mortar_meshes,
      const typename Tags::MortarSize<volume_dim>::type& mortar_sizes,
      const ::dg::Formulation dg_formulation,
      const DirectionMap<
          volume_dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
          face_normal_covector_and_magnitude,
      const TimeStepperType& time_stepper,
      const typename system::boundary_correction_base& boundary_correction,
      const TimeDelta& time_step,
      const Scalar<DataVector>& gts_det_inv_jacobian,
      const VolumeArgs&... volume_args) {
    apply_impl(vars_to_update, mortar_data, volume_mesh, mortar_meshes,
               mortar_sizes, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, time_step,
               std::numeric_limits<double>::signaling_NaN(),
               gts_det_inv_jacobian, volume_args...);
  }

  template <typename... VolumeArgs>
  static void apply(
      const gsl::not_null<typename tag_to_update::type*> vars_to_update,
      const gsl::not_null<MortarDataType*> mortar_data,
      const Mesh<volume_dim>& volume_mesh,
      const typename Tags::MortarMesh<volume_dim>::type& mortar_meshes,
      const typename Tags::MortarSize<volume_dim>::type& mortar_sizes,
      const ::dg::Formulation dg_formulation,
      const DirectionMap<
          volume_dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
          face_normal_covector_and_magnitude,
      const TimeStepperType& time_stepper,
      const typename system::boundary_correction_base& boundary_correction,
      const TimeDelta& time_step, const VolumeArgs&... volume_args) {
    apply_impl(vars_to_update, mortar_data, volume_mesh, mortar_meshes,
               mortar_sizes, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, time_step,
               std::numeric_limits<double>::signaling_NaN(), {},
               volume_args...);
  }

  // dense output (LTS only)
  template <typename... VolumeArgs>
  static void apply(
      const gsl::not_null<typename variables_tag::type*> vars_to_update,
      const MortarDataType& mortar_data, const Mesh<volume_dim>& volume_mesh,
      const typename Tags::MortarMesh<volume_dim>::type& mortar_meshes,
      const typename Tags::MortarSize<volume_dim>::type& mortar_sizes,
      const ::dg::Formulation dg_formulation,
      const DirectionMap<
          volume_dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
          face_normal_covector_and_magnitude,
      const LtsTimeStepper& time_stepper,
      const typename system::boundary_correction_base& boundary_correction,
      const double dense_output_time, const VolumeArgs&... volume_args) {
    apply_impl(vars_to_update, &mortar_data, volume_mesh, mortar_meshes,
               mortar_sizes, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, TimeDelta{},
               dense_output_time, {}, volume_args...);
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ParallelComponent>
  static bool is_ready(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*component*/) {
    if constexpr (local_time_stepping) {
      return receive_boundary_data_local_time_stepping<
          Parallel::is_dg_element_collection_v<ParallelComponent>, System,
          VolumeDim, DenseOutput>(box, inboxes);
    } else {
      return receive_boundary_data_global_time_stepping<
          Parallel::is_dg_element_collection_v<ParallelComponent>,
          Metavariables>(box, inboxes);
    }
  }

 private:
  template <typename... VolumeArgs>
  static void apply_impl(
      const gsl::not_null<typename tag_to_update::type*> vars_to_update,
      const gsl::not_null<MortarDataType*> mortar_data,
      const Mesh<volume_dim>& volume_mesh,
      const typename Tags::MortarMesh<volume_dim>::type& mortar_meshes,
      const typename Tags::MortarSize<volume_dim>::type& mortar_sizes,
      const ::dg::Formulation dg_formulation,
      const DirectionMap<
          volume_dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
          face_normal_covector_and_magnitude,
      const TimeStepperType& time_stepper,
      const typename system::boundary_correction_base& boundary_correction,
      const TimeDelta& time_step, const double dense_output_time,
      const Scalar<DataVector>& gts_det_inv_jacobian,
      const VolumeArgs&... volume_args) {
    tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
        detail::TemporaryReference, volume_tags_for_dg_boundary_terms>>
        volume_args_tuple{volume_args...};

    // Set up helper lambda that will compute and lift the boundary corrections
    ASSERT(
        volume_mesh.quadrature() ==
            make_array<volume_dim>(volume_mesh.quadrature(0)),
        "Must have isotropic quadrature, but got volume mesh: " << volume_mesh);
    const bool using_gauss_lobatto_points =
        volume_mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto;

    Scalar<DataVector> volume_det_inv_jacobian{};
    Scalar<DataVector> volume_det_jacobian{};
    if constexpr (not local_time_stepping) {
      if (not using_gauss_lobatto_points) {
        get(volume_det_inv_jacobian)
            .set_data_ref(make_not_null(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                &const_cast<DataVector&>(get(gts_det_inv_jacobian))));
        get(volume_det_jacobian) = 1.0 / get(volume_det_inv_jacobian);
      }
    }

    using derived_boundary_corrections =
        typename std::decay_t<decltype(boundary_correction)>::creatable_classes;

    static_assert(
        tmpl::all<derived_boundary_corrections, std::is_final<tmpl::_1>>::value,
        "All createable classes for boundary corrections must be marked "
        "final.");
    call_with_dynamic_type<void, derived_boundary_corrections>(
        &boundary_correction,
        [&dense_output_time, &dg_formulation,
         &face_normal_covector_and_magnitude, &mortar_data, &mortar_meshes,
         &mortar_sizes, &time_step, &time_stepper, using_gauss_lobatto_points,
         &vars_to_update, &volume_args_tuple, &volume_det_jacobian,
         &volume_det_inv_jacobian,
         &volume_mesh](auto* typed_boundary_correction) {
          using BcType = std::decay_t<decltype(*typed_boundary_correction)>;
          // Compute internal boundary quantities on the mortar for sides of
          // the element that have neighbors, i.e. they are not an external
          // side.
          using mortar_tags_list = typename BcType::dg_package_field_tags;

          // Variables for reusing allocations.  The actual values are
          // not reused.
          DtVariables dt_boundary_correction_on_mortar{};
          DtVariables volume_dt_correction{};
          // These variables may change size for each mortar and require
          // a new memory allocation, but they may also happen to need
          // to be the same size twice in a row, in which case holding
          // on to the allocation is a win.
          Scalar<DataVector> face_det_jacobian{};
          Variables<mortar_tags_list> local_data_on_mortar{};
          Variables<mortar_tags_list> neighbor_data_on_mortar{};

          for (auto& mortar_id_and_data : *mortar_data) {
            const auto& mortar_id = mortar_id_and_data.first;
            const auto& direction = mortar_id.direction();
            if (UNLIKELY(mortar_id.id() ==
                         ElementId<volume_dim>::external_boundary_id())) {
              ERROR(
                  "Cannot impose boundary conditions on external boundary in "
                  "direction "
                  << direction
                  << " in the ApplyBoundaryCorrections action. Boundary "
                     "conditions are applied in the ComputeTimeDerivative "
                     "action "
                     "instead. You may have unintentionally added external "
                     "mortars in one of the initialization actions.");
            }

            const Mesh<volume_dim - 1> face_mesh =
                volume_mesh.slice_away(direction.dimension());

            const auto compute_correction_coupling =
                [&typed_boundary_correction, &direction, dg_formulation,
                 &dt_boundary_correction_on_mortar, &face_det_jacobian,
                 &face_mesh, &face_normal_covector_and_magnitude,
                 &local_data_on_mortar, &mortar_id, &mortar_meshes,
                 &mortar_sizes, &neighbor_data_on_mortar,
                 using_gauss_lobatto_points, &volume_args_tuple,
                 &volume_det_jacobian, &volume_det_inv_jacobian,
                 &volume_dt_correction, &volume_mesh](
                    const MortarData<volume_dim>& local_mortar_data,
                    const MortarData<volume_dim>& neighbor_mortar_data)
                -> DtVariables {
              if (local_time_stepping and not using_gauss_lobatto_points) {
                // This needs to be updated every call because the Jacobian
                // may be time-dependent. In the case of time-independent maps
                // and local time stepping we could first perform the integral
                // on the boundaries, and then lift to the volume. This is
                // left as a future optimization.
                volume_det_inv_jacobian =
                    local_mortar_data.volume_det_inv_jacobian.value();
                get(volume_det_jacobian) = 1.0 / get(volume_det_inv_jacobian);
              }
              const auto& mortar_mesh = mortar_meshes.at(mortar_id);

              // Extract local and neighbor data, copy into Variables because
              // we store them in a std::vector for type erasure.
              const DataVector& local_data = *local_mortar_data.mortar_data;
              const DataVector& neighbor_data =
                  *neighbor_mortar_data.mortar_data;
              local_data_on_mortar.set_data_ref(
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                  const_cast<double*>(local_data.data()), local_data.size());
              neighbor_data_on_mortar.set_data_ref(
                  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                  const_cast<double*>(neighbor_data.data()),
                  neighbor_data.size());

              // The boundary computations and lifting can be further
              // optimized by in the h-refinement case having only one
              // allocation for the face and having the projection from the
              // mortar to the face be done in place. E.g.
              // local_data_on_mortar and neighbor_data_on_mortar could be
              // allocated fewer times, as well as `needs_projection` section
              // below could do an in-place projection.
              dt_boundary_correction_on_mortar.initialize(
                  mortar_mesh.number_of_grid_points());

              call_boundary_correction(
                  make_not_null(&dt_boundary_correction_on_mortar),
                  local_data_on_mortar, neighbor_data_on_mortar,
                  *typed_boundary_correction, dg_formulation, volume_args_tuple,
                  typename BcType::dg_boundary_terms_volume_tags{});

              const std::array<Spectral::MortarSize, volume_dim - 1>&
                  mortar_size = mortar_sizes.at(mortar_id);

              // This cannot reuse an allocation because it is initialized
              // via move-assignment.  (If it is used at all.)
              DtVariables dt_boundary_correction_projected_onto_face{};
              auto& dt_boundary_correction =
                  [&dt_boundary_correction_on_mortar,
                   &dt_boundary_correction_projected_onto_face, &face_mesh,
                   &mortar_mesh, &mortar_size]() -> DtVariables& {
                if (Spectral::needs_projection(face_mesh, mortar_mesh,
                                               mortar_size)) {
                  dt_boundary_correction_projected_onto_face =
                      ::dg::project_from_mortar(
                          dt_boundary_correction_on_mortar, face_mesh,
                          mortar_mesh, mortar_size);
                  return dt_boundary_correction_projected_onto_face;
                }
                return dt_boundary_correction_on_mortar;
              }();

              // Both paths initialize this to be non-owning.
              Scalar<DataVector> magnitude_of_face_normal{};
              if constexpr (local_time_stepping) {
                (void)face_normal_covector_and_magnitude;
                get(magnitude_of_face_normal)
                    .set_data_ref(make_not_null(&const_cast<DataVector&>(
                        get(local_mortar_data.face_normal_magnitude.value()))));
              } else {
                ASSERT(
                    face_normal_covector_and_magnitude.count(direction) == 1 and
                        face_normal_covector_and_magnitude.at(direction)
                            .has_value(),
                    "Face normal covector and magnitude not set in "
                    "direction: "
                        << direction);
                get(magnitude_of_face_normal)
                    .set_data_ref(make_not_null(&const_cast<DataVector&>(
                        get(get<evolution::dg::Tags::MagnitudeOfNormal>(
                            *face_normal_covector_and_magnitude.at(
                                direction))))));
              }

              if (using_gauss_lobatto_points) {
                // The lift_flux function lifts only on the slice, it does not
                // add the contribution to the volume.
                ::dg::lift_flux(make_not_null(&dt_boundary_correction),
                                volume_mesh.extents(direction.dimension()),
                                magnitude_of_face_normal);
                return std::move(dt_boundary_correction);
              } else {
                // We are using Gauss points.
                //
                // Notes:
                // - We should really lift both sides simultaneously since this
                //   reduces memory accesses. Lifting all sides at the same
                //   time is unlikely to improve performance since we lift by
                //   jumping through slices. There may also be compatibility
                //   issues with local time stepping.
                // - If we lift both sides at the same time we first need to
                //   deal with projecting from mortars to the face, then lift
                //   off the faces. With non-owning Variables memory
                //   allocations could be significantly reduced in this code.
                if constexpr (local_time_stepping) {
                  ASSERT(get(volume_det_inv_jacobian).size() > 0,
                         "For local time stepping the volume determinant of "
                         "the inverse Jacobian has not been set.");

                  get(face_det_jacobian)
                      .set_data_ref(make_not_null(&const_cast<DataVector&>(
                          get(local_mortar_data.face_det_jacobian.value()))));
                } else {
                  // Project the determinant of the Jacobian to the face. This
                  // could be optimized by caching in the time-independent case.
                  get(face_det_jacobian)
                      .destructive_resize(face_mesh.number_of_grid_points());
                  const Matrix identity{};
                  auto interpolation_matrices =
                      make_array<volume_dim>(std::cref(identity));
                  const std::pair<Matrix, Matrix>& matrices =
                      Spectral::boundary_interpolation_matrices(
                          volume_mesh.slice_through(direction.dimension()));
                  gsl::at(interpolation_matrices, direction.dimension()) =
                      direction.side() == Side::Upper ? matrices.second
                                                      : matrices.first;
                  apply_matrices(make_not_null(&get(face_det_jacobian)),
                                 interpolation_matrices,
                                 get(volume_det_jacobian),
                                 volume_mesh.extents());
                }

                volume_dt_correction.initialize(
                    volume_mesh.number_of_grid_points(), 0.0);
                ::dg::lift_boundary_terms_gauss_points(
                    make_not_null(&volume_dt_correction),
                    volume_det_inv_jacobian, volume_mesh, direction,
                    dt_boundary_correction, magnitude_of_face_normal,
                    face_det_jacobian);
                return std::move(volume_dt_correction);
              }
            };

            if constexpr (local_time_stepping) {
              typename variables_tag::type lgl_lifted_data{};
              auto& lifted_data = using_gauss_lobatto_points ? lgl_lifted_data
                                                             : *vars_to_update;
              if (using_gauss_lobatto_points) {
                lifted_data.initialize(face_mesh.number_of_grid_points(), 0.0);
              }

              auto& mortar_data_history = mortar_id_and_data.second;
              if constexpr (DenseOutput) {
                (void)time_step;
                time_stepper.boundary_dense_output(
                    &lifted_data, mortar_data_history, dense_output_time,
                    compute_correction_coupling);
              } else {
                (void)dense_output_time;
                time_stepper.add_boundary_delta(&lifted_data,
                                                mortar_data_history, time_step,
                                                compute_correction_coupling);
              }

              if (using_gauss_lobatto_points) {
                // Add the flux contribution to the volume data
                add_slice_to_data(
                    vars_to_update, lifted_data, volume_mesh.extents(),
                    direction.dimension(),
                    index_to_slice_at(volume_mesh.extents(), direction));
              }
            } else {
              (void)time_step;
              (void)time_stepper;
              (void)dense_output_time;

              // Choose an allocation cache that may be empty, so we
              // might be able to reuse the allocation obtained for the
              // lifted data.  This may result in a self assignment,
              // depending on the code paths taken, but handling the
              // results this way makes the GTS and LTS paths more
              // similar because the LTS code always stores the result
              // in the history and so sometimes benefits from moving
              // into the return value of compute_correction_coupling.
              auto& lifted_data = using_gauss_lobatto_points
                                      ? dt_boundary_correction_on_mortar
                                      : volume_dt_correction;
              lifted_data = compute_correction_coupling(
                  mortar_id_and_data.second.local(),
                  mortar_id_and_data.second.neighbor());

              if (using_gauss_lobatto_points) {
                // Add the flux contribution to the volume data
                add_slice_to_data(
                    vars_to_update, lifted_data, volume_mesh.extents(),
                    direction.dimension(),
                    index_to_slice_at(volume_mesh.extents(), direction));
              } else {
                *vars_to_update += lifted_data;
              }
            }
          }
        });
  }

  template <typename... BoundaryCorrectionTags, typename... Tags,
            typename BoundaryCorrection, typename... AllVolumeArgs,
            typename... VolumeTagsForCorrection>
  static void call_boundary_correction(
      const gsl::not_null<Variables<tmpl::list<BoundaryCorrectionTags...>>*>
          boundary_corrections_on_mortar,
      const Variables<tmpl::list<Tags...>>& local_boundary_data,
      const Variables<tmpl::list<Tags...>>& neighbor_boundary_data,
      const BoundaryCorrection& boundary_correction,
      const ::dg::Formulation dg_formulation,
      const tuples::TaggedTuple<detail::TemporaryReference<AllVolumeArgs>...>&
          volume_args_tuple,
      tmpl::list<VolumeTagsForCorrection...> /*meta*/) {
    boundary_correction.dg_boundary_terms(
        make_not_null(
            &get<BoundaryCorrectionTags>(*boundary_corrections_on_mortar))...,
        get<Tags>(local_boundary_data)..., get<Tags>(neighbor_boundary_data)...,
        dg_formulation,
        tuples::get<detail::TemporaryReference<VolumeTagsForCorrection>>(
            volume_args_tuple)...);
  }
};

namespace Actions {
/*!
 * \brief Computes the boundary corrections for global time-stepping
 * and adds them to the time derivative.
 */
template <typename System, size_t VolumeDim, bool DenseOutput,
          bool UseNodegroupDgElements>
struct ApplyBoundaryCorrectionsToTimeDerivative {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          VolumeDim, UseNodegroupDgElements>>;
  using const_global_cache_tags =
      tmpl::list<evolution::Tags::BoundaryCorrection<System>,
                 ::dg::Tags::Formulation>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        UseNodegroupDgElements ==
            Parallel::is_dg_element_collection_v<ParallelComponent>,
        "The action ApplyBoundaryCorrectionsToTimeDerivative is told by the "
        "template parameter UseNodegroupDgElements that it is being "
        "used with a DgElementCollection, but the ParallelComponent "
        "is not a DgElementCollection. You need to change the template "
        "parameter on the ApplyBoundaryCorrectionsToTimeDerivative action "
        "in your action list.");
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    const Element<volume_dim>& element =
        db::get<domain::Tags::Element<volume_dim>>(box);

    if (UNLIKELY(element.number_of_neighbors() == 0)) {
      // We have no neighbors, yay!
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    if (not receive_boundary_data_global_time_stepping<
            Parallel::is_dg_element_collection_v<ParallelComponent>,
            Metavariables>(make_not_null(&box), make_not_null(&inboxes))) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    db::mutate_apply<
        ApplyBoundaryCorrections<false, System, VolumeDim, DenseOutput>>(
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Computes the boundary corrections for local time-stepping
 * and adds them to the variables.
 *
 * When using local time stepping the neighbor sends data at the neighbor's
 * current temporal id. Along with the boundary data, the next temporal id at
 * which the neighbor will send data is also sent. This is equal to the
 * neighbor's `::Tags::Next<::Tags::TimeStepId>`. When inserting into the mortar
 * data history, we insert the received temporal id, that is, the current time
 * of the neighbor, along with the boundary correction data.
 */
template <typename System, size_t VolumeDim, bool DenseOutput,
          bool UseNodegroupDgElements>
struct ApplyLtsBoundaryCorrections {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          VolumeDim, UseNodegroupDgElements>>;
  using const_global_cache_tags =
      tmpl::list<evolution::Tags::BoundaryCorrection<System>,
                 ::dg::Tags::Formulation>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        UseNodegroupDgElements ==
            Parallel::is_dg_element_collection_v<ParallelComponent>,
        "The action ApplyLtsBoundaryCorrections is told by the "
        "template parameter UseNodegroupDgElements that it is being "
        "used with a DgElementCollection, but the ParallelComponent "
        "is not a DgElementCollection. You need to change the "
        "template parameter on the ApplyLtsBoundaryCorrections action "
        "in your action list.");
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    const Element<volume_dim>& element =
        db::get<domain::Tags::Element<volume_dim>>(box);

    if (UNLIKELY(element.number_of_neighbors() == 0)) {
      // We have no neighbors, yay!
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    if (not receive_boundary_data_local_time_stepping<
            Parallel::is_dg_element_collection_v<ParallelComponent>, System,
            VolumeDim, false>(make_not_null(&box), make_not_null(&inboxes))) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    if (::SelfStart::step_unused(
            db::get<::Tags::TimeStepId>(box),
            db::get<::Tags::Next<::Tags::TimeStepId>>(box))) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    db::mutate_apply<
        ApplyBoundaryCorrections<true, System, VolumeDim, DenseOutput>>(
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace evolution::dg
