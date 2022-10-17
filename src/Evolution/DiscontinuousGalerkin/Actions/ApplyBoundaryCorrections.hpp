// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/TrimMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Tags/NeighborMesh.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Tags.hpp"
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
namespace evolution::dg::subcell {
// We use a forward declaration instead of including a header file to avoid
// coupling to the DG-subcell libraries for executables that don't use subcell.
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
                       std::optional<std::vector<double>>,
                       std::optional<std::vector<double>>, ::TimeStepId>,
            boost::hash<std::pair<Direction<Metavariables::volume_dim>,
                                  ElementId<Metavariables::volume_dim>>>>>*>
        received_temporal_id_and_data);
}  // namespace evolution::dg::subcell
/// \endcond

namespace evolution::dg {
/// Receive boundary data for global time-stepping.  Returns true if
/// all necessary data has been received.
template <typename Metavariables, typename DbTagsList, typename... InboxTags>
bool receive_boundary_data_global_time_stepping(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes) {
  constexpr size_t volume_dim = Metavariables::system::volume_dim;

  const TimeStepId& temporal_id = get<::Tags::TimeStepId>(*box);
  using Key = std::pair<Direction<volume_dim>, ElementId<volume_dim>>;
  std::map<
      TimeStepId,
      FixedHashMap<maximum_number_of_neighbors(volume_dim), Key,
                   std::tuple<Mesh<volume_dim>, Mesh<volume_dim - 1>,
                              std::optional<std::vector<double>>,
                              std::optional<std::vector<double>>, ::TimeStepId>,
                   boost::hash<Key>>>& inbox =
      tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          volume_dim>>(*inboxes);
  const auto received_temporal_id_and_data = inbox.find(temporal_id);
  if (received_temporal_id_and_data == inbox.end()) {
    return false;
  }
  const auto& received_neighbor_data = received_temporal_id_and_data->second;
  const Element<volume_dim>& element =
      db::get<domain::Tags::Element<volume_dim>>(*box);
  for (const auto& [direction, neighbors] : element.neighbors()) {
    for (const auto& neighbor : neighbors) {
      const auto neighbor_received =
          received_neighbor_data.find(Key{direction, neighbor});
      if (neighbor_received == received_neighbor_data.end()) {
        return false;
      }
    }
  }

  // Move inbox contents into the DataBox
  if constexpr (using_subcell_v<Metavariables>) {
    evolution::dg::subcell::neighbor_reconstructed_face_solution<Metavariables>(
        box, make_not_null(&*received_temporal_id_and_data));
  }

  db::mutate<evolution::dg::Tags::MortarData<volume_dim>,
             evolution::dg::Tags::MortarNextTemporalId<volume_dim>,
             evolution::dg::Tags::NeighborMesh<volume_dim>>(
      box,
      [&received_temporal_id_and_data](
          const gsl::not_null<std::unordered_map<
              Key, evolution::dg::MortarData<volume_dim>, boost::hash<Key>>*>
              mortar_data,
          const gsl::not_null<
              std::unordered_map<Key, TimeStepId, boost::hash<Key>>*>
              mortar_next_time_step_id,
          const gsl::not_null<FixedHashMap<
              maximum_number_of_neighbors(volume_dim),
              std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
              Mesh<volume_dim>,
              boost::hash<
                  std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>*>
              neighbor_mesh) {
        neighbor_mesh->clear();
        for (auto& received_mortar_data :
             received_temporal_id_and_data->second) {
          const auto& mortar_id = received_mortar_data.first;
          ASSERT(received_temporal_id_and_data->first ==
                     mortar_data->at(mortar_id).time_step_id(),
                 "Expected to receive mortar data on mortar "
                     << mortar_id << " at time "
                     << mortar_next_time_step_id->at(mortar_id)
                     << " but actually received at time "
                     << received_temporal_id_and_data->first);
          neighbor_mesh->insert_or_assign(
              mortar_id, std::get<0>(received_mortar_data.second));
          mortar_next_time_step_id->at(mortar_id) =
              std::get<4>(received_mortar_data.second);
          ASSERT(using_subcell_v<Metavariables> or
                     std::get<3>(received_mortar_data.second).has_value(),
                 "Must receive number boundary correction data when not using "
                 "DG-subcell.");
          if (std::get<3>(received_mortar_data.second).has_value()) {
            mortar_data->at(mortar_id).insert_neighbor_mortar_data(
                received_temporal_id_and_data->first,
                std::get<1>(received_mortar_data.second),
                std::move(*std::get<3>(received_mortar_data.second)));
          }
        }
      });
  inbox.erase(received_temporal_id_and_data);
  return true;
}

/// Receive boundary data for local time-stepping.  Returns true if
/// all necessary data has been received.
///
/// Setting \p DenseOutput to true receives data required for output
/// at `::Tags::Time` instead of `::Tags::Next<::Tags::TimeStepId>`.
template <typename Metavariables, bool DenseOutput = false, typename DbTagsList,
          typename... InboxTags>
bool receive_boundary_data_local_time_stepping(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes) {
  constexpr size_t volume_dim = Metavariables::system::volume_dim;

  using variables_tag = typename Metavariables::system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  // The boundary history coupling computation (which computes the _lifted_
  // boundary correction) returns a Variables<dt<EvolvedVars>> instead of
  // using the `NormalDotNumericalFlux` prefix tag. This is because the
  // returned quantity is more a `dt` quantity than a
  // `NormalDotNormalDotFlux` since it's been lifted to the volume.
  using Key = std::pair<Direction<volume_dim>, ElementId<volume_dim>>;
  std::map<
      TimeStepId,
      FixedHashMap<maximum_number_of_neighbors(volume_dim), Key,
                   std::tuple<Mesh<volume_dim>, Mesh<volume_dim - 1>,
                              std::optional<std::vector<double>>,
                              std::optional<std::vector<double>>, ::TimeStepId>,
                   boost::hash<Key>>>& inbox =
      tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          volume_dim>>(*inboxes);

  const auto needed_time = [&box]() {
    const auto& local_next_temporal_id =
        db::get<::Tags::Next<::Tags::TimeStepId>>(*box);
    if constexpr (DenseOutput) {
      const auto& dense_output_time = db::get<::Tags::Time>(*box);
      return [&dense_output_time,
              &local_next_temporal_id](const TimeStepId& id) {
        return evolution_less<double>{
            local_next_temporal_id.time_runs_forward()}(id.step_time().value(),
                                                        dense_output_time);
      };
    } else {
      return [&local_next_temporal_id](const TimeStepId& id) {
        return id < local_next_temporal_id;
      };
    }
  }();

  const bool have_all_intermediate_messages =
      db::mutate<evolution::dg::Tags::MortarDataHistory<
                     volume_dim, typename dt_variables_tag::type>,
                 evolution::dg::Tags::MortarNextTemporalId<volume_dim>,
                 evolution::dg::Tags::NeighborMesh<volume_dim>>(
          box,
          [&inbox, &needed_time](
              const gsl::not_null<
                  std::unordered_map<Key,
                                     TimeSteppers::BoundaryHistory<
                                         evolution::dg::MortarData<volume_dim>,
                                         evolution::dg::MortarData<volume_dim>,
                                         typename dt_variables_tag::type>,
                                     boost::hash<Key>>*>
                  boundary_data_history,
              const gsl::not_null<
                  std::unordered_map<Key, TimeStepId, boost::hash<Key>>*>
                  mortar_next_time_step_id,
              const gsl::not_null<FixedHashMap<
                  maximum_number_of_neighbors(volume_dim),
                  std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
                  Mesh<volume_dim>,
                  boost::hash<std::pair<Direction<volume_dim>,
                                        ElementId<volume_dim>>>>*>
                  neighbor_mesh,
              const Element<volume_dim>& element) {
            // Remove neighbor meshes for neighbors that don't exist anymore
            domain::remove_nonexistent_neighbors(neighbor_mesh, element);

            // Move received boundary data into boundary history.
            for (auto received_data = inbox.begin();
                 received_data != inbox.end() and
                 needed_time(received_data->first);
                 received_data = inbox.erase(received_data)) {
              const auto& receive_temporal_id = received_data->first;
              // Loop over all mortars for which we received data at this time
              for (auto received_mortar_data = received_data->second.begin();
                   received_mortar_data != received_data->second.end();
                   received_mortar_data =
                       received_data->second.erase(received_mortar_data)) {
                const auto& mortar_id = received_mortar_data->first;
                MortarData<Metavariables::volume_dim> neighbor_mortar_data{};
                // Insert:
                // - the current TimeStepId of the neighbor
                // - the current face mesh of the neighbor
                // - the current boundary correction data of the neighbor
                ASSERT(std::get<3>(received_mortar_data->second).has_value(),
                       "Did not receive boundary correction data from the "
                       "neighbor\nMortarId: "
                           << mortar_id
                           << "\nTimeStepId: " << receive_temporal_id);
                ASSERT(mortar_next_time_step_id->at(mortar_id) >=
                           receive_temporal_id,
                       "Expected to receive mortar data on mortar "
                           << mortar_id << " at time "
                           << mortar_next_time_step_id->at(mortar_id)
                           << " but actually received at time "
                           << receive_temporal_id);
                if (mortar_next_time_step_id->at(mortar_id) !=
                    receive_temporal_id) {
                  // We've received messages from our neighbor
                  // out-of-order.  They are always sent in-order, but
                  // messages are not guaranteed to be received in the
                  // order they were sent.
                  return false;
                }
                neighbor_mesh->insert_or_assign(
                    mortar_id, std::get<0>(received_mortar_data->second));
                mortar_next_time_step_id->at(mortar_id) =
                    std::get<4>(received_mortar_data->second);
                neighbor_mortar_data.insert_neighbor_mortar_data(
                    receive_temporal_id,
                    std::get<1>(received_mortar_data->second),
                    std::move(*std::get<3>(received_mortar_data->second)));
                boundary_data_history->at(mortar_id).remote_insert(
                    receive_temporal_id, std::move(neighbor_mortar_data));
              }
            }
            return true;
          },
          db::get<::domain::Tags::Element<volume_dim>>(*box));

  if (not have_all_intermediate_messages) {
    return false;
  }

  return alg::all_of(
      db::get<evolution::dg::Tags::MortarNextTemporalId<volume_dim>>(*box),
      [&needed_time](
          const std::pair<Key, TimeStepId>& mortar_id_and_next_temporal_id) {
        return mortar_id_and_next_temporal_id.first.second ==
                   ElementId<volume_dim>::external_boundary_id() or
               not needed_time(mortar_id_and_next_temporal_id.second);
      });
}

/// Apply corrections from boundary communication.
///
/// If `Metavariables::local_time_stepping` is false, updates the
/// derivative of the variables, which should be done before taking a
/// time step.  If `Metavariables::local_time_stepping` is true,
/// updates the variables themselves, which should be done after the
/// volume update.
///
/// Setting \p DenseOutput to true receives data required for output
/// at ::Tags::Time instead of performing a full step.  This is only
/// used for local time-stepping.
template <typename Metavariables, bool DenseOutput = false>
struct ApplyBoundaryCorrections {
  static constexpr bool local_time_stepping =
      Metavariables::local_time_stepping;
  static_assert(local_time_stepping or not DenseOutput,
                "GTS does not use ApplyBoundaryCorrections for dense output.");

  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = system::volume_dim;
  using variables_tag = typename system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using DtVariables = typename dt_variables_tag::type;

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
  using argument_tags = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<DenseOutput, mortar_data_tag, tmpl::list<>>,
      domain::Tags::Mesh<volume_dim>, Tags::MortarMesh<volume_dim>,
      Tags::MortarSize<volume_dim>, ::dg::Tags::Formulation,
      evolution::dg::Tags::NormalCovectorAndMagnitude<volume_dim>,
      ::Tags::TimeStepper<>, evolution::Tags::BoundaryCorrection<system>,
      tmpl::conditional_t<DenseOutput, ::Tags::Time, ::Tags::TimeStep>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                                       Frame::Inertial>>>>;

  // full step
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
      const Scalar<DataVector>& gts_det_inv_jacobian = {}) {
    apply_impl(vars_to_update, mortar_data, volume_mesh, mortar_meshes,
               mortar_sizes, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, time_step,
               std::numeric_limits<double>::signaling_NaN(),
               gts_det_inv_jacobian);
  }

  // dense output (LTS only)
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
      const double dense_output_time) {
    apply_impl(vars_to_update, &mortar_data, volume_mesh, mortar_meshes,
               mortar_sizes, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, TimeDelta{},
               dense_output_time, {});
  }

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ParallelComponent>
  static bool is_ready(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*component*/) {
    if constexpr (local_time_stepping) {
      return receive_boundary_data_local_time_stepping<Metavariables,
                                                       DenseOutput>(box,
                                                                    inboxes);
    } else {
      return receive_boundary_data_global_time_stepping<Metavariables>(box,
                                                                       inboxes);
    }
  }

 private:
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
      const Scalar<DataVector>& gts_det_inv_jacobian) {
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
         &vars_to_update, &volume_det_jacobian, &volume_det_inv_jacobian,
         &volume_mesh](auto* typed_boundary_correction) {
          // Compute internal boundary quantities on the mortar for sides of
          // the element that have neighbors, i.e. they are not an external
          // side.
          using mortar_tags_list = typename std::decay_t<
              decltype(*typed_boundary_correction)>::dg_package_field_tags;

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
            const auto& direction = mortar_id.first;
            if (UNLIKELY(mortar_id.second ==
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
                 using_gauss_lobatto_points, &volume_det_jacobian,
                 &volume_det_inv_jacobian, &volume_dt_correction, &volume_mesh](
                    const MortarData<volume_dim>& local_mortar_data,
                    const MortarData<volume_dim>& neighbor_mortar_data)
                -> DtVariables {
              if (local_time_stepping and not using_gauss_lobatto_points) {
                // This needs to be updated every call because the Jacobian
                // may be time-dependent. In the case of time-independent maps
                // and local time stepping we could first perform the integral
                // on the boundaries, and then lift to the volume. This is
                // left as a future optimization.
                local_mortar_data.get_local_volume_det_inv_jacobian(
                    make_not_null(&volume_det_inv_jacobian));
                get(volume_det_jacobian) = 1.0 / get(volume_det_inv_jacobian);
              }
              const auto& mortar_mesh = mortar_meshes.at(mortar_id);

              // Extract local and neighbor data, copy into Variables because
              // we store them in a std::vector for type erasure.
              const std::pair<Mesh<volume_dim - 1>, std::vector<double>>&
                  local_mesh_and_data = *local_mortar_data.local_mortar_data();
              const std::pair<Mesh<volume_dim - 1>, std::vector<double>>&
                  neighbor_mesh_and_data =
                      *neighbor_mortar_data.neighbor_mortar_data();
              local_data_on_mortar.initialize(
                  mortar_mesh.number_of_grid_points());
              neighbor_data_on_mortar.initialize(
                  mortar_mesh.number_of_grid_points());
              std::copy(std::get<1>(local_mesh_and_data).begin(),
                        std::get<1>(local_mesh_and_data).end(),
                        local_data_on_mortar.data());
              std::copy(std::get<1>(neighbor_mesh_and_data).begin(),
                        std::get<1>(neighbor_mesh_and_data).end(),
                        neighbor_data_on_mortar.data());

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
                  *typed_boundary_correction, dg_formulation);

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
                local_mortar_data.get_local_face_normal_magnitude(
                    &magnitude_of_face_normal);
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

                  local_mortar_data.get_local_face_det_jacobian(
                      make_not_null(&face_det_jacobian));
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
                evolution::dg::lift_boundary_terms_gauss_points(
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
                time_stepper.add_boundary_delta(
                    &lifted_data, make_not_null(&mortar_data_history),
                    time_step, compute_correction_coupling);
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
                  mortar_id_and_data.second, mortar_id_and_data.second);
              // Remove data since it's tagged with the time. In the future we
              // _might_ be able to reuse allocations, but this optimization
              // should only be done after profiling.
              mortar_id_and_data.second.extract();

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
            typename BoundaryCorrection>
  static void call_boundary_correction(
      const gsl::not_null<Variables<tmpl::list<BoundaryCorrectionTags...>>*>
          boundary_corrections_on_mortar,
      const Variables<tmpl::list<Tags...>>& local_boundary_data,
      const Variables<tmpl::list<Tags...>>& neighbor_boundary_data,
      const BoundaryCorrection& boundary_correction,
      const ::dg::Formulation dg_formulation) {
    boundary_correction.dg_boundary_terms(
        make_not_null(
            &get<BoundaryCorrectionTags>(*boundary_corrections_on_mortar))...,
        get<Tags>(local_boundary_data)..., get<Tags>(neighbor_boundary_data)...,
        dg_formulation);
  }
};

namespace Actions {
/*!
 * \brief Computes the boundary corrections for global time-stepping
 * and adds them to the time derivative.
 */
template <typename Metavariables>
struct ApplyBoundaryCorrectionsToTimeDerivative {
  static_assert(not Metavariables::local_time_stepping,
                "LTS boundary corrections must be applied to the solution "
                "after each time step using the ApplyLtsBoundaryCorrections "
                "action.");
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Metavariables::volume_dim>>;
  using const_global_cache_tags = tmpl::list<
      evolution::Tags::BoundaryCorrection<typename Metavariables::system>,
      ::dg::Tags::Formulation>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    const Element<volume_dim>& element =
        db::get<domain::Tags::Element<volume_dim>>(box);

    if (UNLIKELY(element.number_of_neighbors() == 0)) {
      // We have no neighbors, yay!
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    if (not receive_boundary_data_global_time_stepping<Metavariables>(
            make_not_null(&box), make_not_null(&inboxes))) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    db::mutate_apply<ApplyBoundaryCorrections<Metavariables>>(
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
template <typename Metavariables>
struct ApplyLtsBoundaryCorrections {
  static_assert(Metavariables::local_time_stepping,
                "GTS boundary corrections must be applied to the time "
                "derivative before each time step using the "
                "ApplyBoundaryCorrectionsToTimeDerivative action.");
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Metavariables::volume_dim>>;
  using const_global_cache_tags = tmpl::list<
      evolution::Tags::BoundaryCorrection<typename Metavariables::system>,
      ::dg::Tags::Formulation>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    const Element<volume_dim>& element =
        db::get<domain::Tags::Element<volume_dim>>(box);

    if (UNLIKELY(element.number_of_neighbors() == 0)) {
      // We have no neighbors, yay!
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    if (not receive_boundary_data_local_time_stepping<Metavariables>(
            make_not_null(&box), make_not_null(&inboxes))) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    db::mutate_apply<ApplyBoundaryCorrections<Metavariables>>(
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace evolution::dg
