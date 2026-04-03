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
#include "Domain/Structure/Topology.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/BoundaryInterpolationMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
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
void neighbor_reconstructed_face_solution(gsl::not_null<db::Access*> box);
template <size_t Dim>
void neighbor_tci_decision(
    gsl::not_null<db::Access*> box,
    const DirectionalId<Dim>& directional_element_id,
    const evolution::dg::BoundaryData<Dim>& neighbor_data);
template <size_t VolumeDim>
void receive_subcell_data_for_dg(
    gsl::not_null<db::Access*> box, const DirectionalId<VolumeDim>& mortar_id,
    const evolution::dg::BoundaryData<VolumeDim>& received_mortar_data);
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
}  // namespace detail

/// Move boundary data from the inbox to the DataBox.  Returns true if
/// all necessary data has been received.
///
/// Setting \p DenseOutput to true receives data required for output
/// at `::Tags::Time` instead of `::Tags::Next<::Tags::TimeStepId>`.
template <bool UseNodegroupDgElements, typename Metavariables,
          bool LocalTimeStepping, bool DenseOutput, typename DbTagsList,
          typename... InboxTags>
bool receive_boundary_data(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes) {
  constexpr size_t volume_dim = Metavariables::system::volume_dim;
  constexpr size_t face_dim = volume_dim - 1;

  const auto needed_time = [&box]() {
    if constexpr (LocalTimeStepping) {
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
    } else {
      static_assert(not DenseOutput,
                    "Should not be receiving data for dense output with GTS.");
      const auto& current_id = db::get<::Tags::TimeStepId>(*box);
      return [&current_id](const TimeStepId& id) { return id <= current_id; };
    }
  }();

  auto& inbox =
      tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          volume_dim, UseNodegroupDgElements>>(*inboxes);

  const auto& element = db::get<domain::Tags::Element<volume_dim>>(*box);
  const auto& volume_mesh = db::get<domain::Tags::Mesh<volume_dim>>(*box);
  const auto& mortar_infos = db::get<Tags::MortarInfo<volume_dim>>(*box);

  size_t missing_messages{};
  do {
    // The boundary history coupling computation (which computes the _lifted_
    // boundary correction) returns a Variables<dt<EvolvedVars>> instead of
    // using the `NormalDotNumericalFlux` prefix tag. This is because the
    // returned quantity is more a `dt` quantity than a
    // `NormalDotNormalDotFlux` since it's been lifted to the volume.
    using InboxMap = std::map<
        TimeStepId,
        DirectionalIdMap<volume_dim, evolution::dg::BoundaryData<volume_dim>>>;
    inbox.collect_messages();
    InboxMap& inbox_data = inbox.messages;

    missing_messages = 0;

    for (const auto& [direction, neighbors] : element.neighbors()) {
      for (const auto& neighbor : neighbors) {
        const DirectionalId mortar_id{direction, neighbor};

        const auto& time_stepping_policy =
            mortar_infos.at(mortar_id).time_stepping_policy();
        switch (time_stepping_policy) {
          case TimeSteppingPolicy::EqualRate:
            if (LocalTimeStepping) {
              continue;
            }
            break;
          case TimeSteppingPolicy::Conservative:
            if (not LocalTimeStepping) {
              continue;
            }
            break;
          default:
            ERROR("Unhandled TimeSteppingPolicy: " << time_stepping_policy);
        }

        const size_t sliced_away_dim = direction.dimension();
        const Mesh<volume_dim - 1> face_mesh =
            volume_mesh.slice_away(sliced_away_dim);
        for (;;) {
          const auto mortar_next_time_step_id =
              db::get<evolution::dg::Tags::MortarNextTemporalId<volume_dim>>(
                  *box)
                  .at(mortar_id);
          if (not needed_time(mortar_next_time_step_id)) {
            break;
          }
          const auto time_entry = inbox_data.find(mortar_next_time_step_id);
          if (time_entry == inbox_data.end()) {
            ++missing_messages;
            break;
          }
          const auto received_mortar_data = time_entry->second.find(mortar_id);
          if (received_mortar_data == time_entry->second.end()) {
            ++missing_messages;
            break;
          }

          if constexpr (using_subcell_v<Metavariables>) {
            if (time_stepping_policy == TimeSteppingPolicy::EqualRate) {
              evolution::dg::subcell::receive_subcell_data_for_dg<volume_dim>(
                  &db::as_access(*box), mortar_id,
                  received_mortar_data->second);
              evolution::dg::subcell::neighbor_tci_decision<volume_dim>(
                  make_not_null(&db::as_access(*box)), mortar_id,
                  received_mortar_data->second);
            }
          }

          db::mutate<evolution::dg::Tags::MortarMesh<volume_dim>,
                     evolution::dg::Tags::MortarData<volume_dim>,
                     evolution::dg::Tags::MortarDataHistory<volume_dim>,
                     evolution::dg::Tags::MortarNextTemporalId<volume_dim>,
                     domain::Tags::NeighborMesh<volume_dim>>(
              [&](const gsl::not_null<
                      DirectionalIdMap<volume_dim, Mesh<volume_dim - 1>>*>
                      mortar_meshes,
                  const gsl::not_null<DirectionalIdMap<
                      volume_dim, evolution::dg::MortarDataHolder<volume_dim>>*>
                      gts_mortar_data,
                  const gsl::not_null<DirectionalIdMap<
                      volume_dim,
                      TimeSteppers::BoundaryHistory<
                          evolution::dg::MortarData<volume_dim>,
                          evolution::dg::MortarData<volume_dim>, DataVector>>*>
                      boundary_data_history,
                  const gsl::not_null<DirectionalIdMap<volume_dim, TimeStepId>*>
                      mortar_next_time_step_ids,
                  const gsl::not_null<
                      DirectionalIdMap<volume_dim, Mesh<volume_dim>>*>
                      neighbor_mesh) {
                const Mesh<face_dim> neighbor_face_mesh =
                    received_mortar_data->second.volume_mesh.slice_away(
                        sliced_away_dim);
                const Mesh<face_dim> mortar_mesh =
                    ::dg::mortar_mesh(face_mesh, neighbor_face_mesh);

                const auto project_boundary_mortar_data =
                    [&mortar_mesh](const TimeStepId& /*id*/,
                                   const gsl::not_null<
                                       ::evolution::dg::MortarData<volume_dim>*>
                                       mortar_data) {
                      return p_project_mortar_data(mortar_data, mortar_mesh);
                    };

                mortar_meshes->at(mortar_id) = mortar_mesh;
                switch (time_stepping_policy) {
                  case TimeSteppingPolicy::EqualRate:
                    p_project_mortar_data(
                        make_not_null(&gts_mortar_data->at(mortar_id).local()),
                        mortar_mesh);
                    break;
                  case TimeSteppingPolicy::Conservative:
                    boundary_data_history->at(mortar_id).local().for_each(
                        project_boundary_mortar_data);
                    break;
                  default:
                    ERROR("Unhandled TimeSteppingPolicy: "
                          << time_stepping_policy);
                }

                neighbor_mesh->insert_or_assign(
                    mortar_id, received_mortar_data->second.volume_mesh);
                mortar_next_time_step_ids->at(mortar_id) =
                    received_mortar_data->second.validity_range;

                ASSERT(using_subcell_v<Metavariables> or
                           received_mortar_data->second.boundary_correction_data
                               .has_value(),
                       "Must receive neighbor boundary correction data when "
                       "not using DG-subcell. Mortar ID is: ("
                           << mortar_id.direction() << "," << mortar_id.id()
                           << ") and TimeStepId is " << time_entry->first);
                MortarData<volume_dim> neighbor_mortar_data{};
                neighbor_mortar_data.face_mesh = neighbor_face_mesh;
                neighbor_mortar_data.mortar_mesh =
                    received_mortar_data->second.boundary_correction_mesh;
                neighbor_mortar_data.mortar_data = std::move(
                    received_mortar_data->second.boundary_correction_data);
                switch (time_stepping_policy) {
                  case TimeSteppingPolicy::EqualRate:
                    if (neighbor_mortar_data.mortar_data.has_value()) {
                      p_project_mortar_data(
                          make_not_null(&neighbor_mortar_data), mortar_mesh);
                    }
                    gts_mortar_data->at(mortar_id).neighbor() =
                        std::move(neighbor_mortar_data);
                    break;
                  case TimeSteppingPolicy::Conservative:
                    ASSERT(neighbor_mortar_data.mortar_data.has_value(),
                           "Did not receive mortar data for " << mortar_id);
                    boundary_data_history->at(mortar_id).remote().insert(
                        time_entry->first,
                        received_mortar_data->second.integration_order,
                        std::move(neighbor_mortar_data));
                    boundary_data_history->at(mortar_id).remote().for_each(
                        project_boundary_mortar_data);
                    break;
                  default:
                    ERROR("Unhandled TimeSteppingPolicy: "
                          << time_stepping_policy);
                }
              },
              box);
          time_entry->second.erase(received_mortar_data);
          if (time_entry->second.empty()) {
            inbox_data.erase(time_entry);
          }
        }
      }
    }

    if (missing_messages == 0) {
      if constexpr (using_subcell_v<Metavariables>) {
        evolution::dg::subcell::neighbor_reconstructed_face_solution<
            volume_dim, typename Metavariables::SubcellOptions::
                            DgComputeSubcellNeighborPackagedData>(
            &db::as_access(*box));
      }
      return true;
    }
  } while (inbox.set_missing_messages(missing_messages));
  return false;
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
template <bool LocalTimeStepping, typename Metavariables, size_t VolumeDim,
          bool DenseOutput>
struct ApplyBoundaryCorrections {
  static constexpr bool local_time_stepping = LocalTimeStepping;
  static_assert(local_time_stepping or not DenseOutput,
                "GTS does not use ApplyBoundaryCorrections for dense output.");

  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = VolumeDim;
  using variables_tag = typename system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using DtVariables = typename dt_variables_tag::type;
  using derived_boundary_corrections =
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               evolution::BoundaryCorrection>;
  using volume_tags_for_dg_boundary_terms = tmpl::remove_duplicates<
      tmpl::flatten<tmpl::transform<derived_boundary_corrections,
                                    detail::get_dg_boundary_terms<tmpl::_1>>>>;

  using TimeStepperType =
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>;

  using tag_to_update =
      tmpl::conditional_t<local_time_stepping, variables_tag, dt_variables_tag>;
  using mortar_data_tag =
      tmpl::conditional_t<local_time_stepping,
                          evolution::dg::Tags::MortarDataHistory<volume_dim>,
                          evolution::dg::Tags::MortarData<volume_dim>>;

  using return_tags = tmpl::list<tag_to_update>;
  using argument_tags = tmpl::append<
      tmpl::flatten<tmpl::list<
          mortar_data_tag, domain::Tags::Mesh<volume_dim>,
          domain::Tags::Element<volume_dim>, Tags::MortarMesh<volume_dim>,
          Tags::MortarInfo<volume_dim>, ::dg::Tags::Formulation,
          evolution::dg::Tags::NormalCovectorAndMagnitude<volume_dim>,
          ::Tags::TimeStepper<TimeStepperType>,
          evolution::Tags::BoundaryCorrection,
          tmpl::conditional_t<DenseOutput, ::Tags::Time, ::Tags::TimeStep>,
          tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                              domain::Tags::DetInvJacobian<
                                  Frame::ElementLogical, Frame::Inertial>>>>,
      volume_tags_for_dg_boundary_terms>;

  // full step
  template <typename... VolumeArgs>
  static void apply(
      const gsl::not_null<typename tag_to_update::type*> vars_to_update,
      const typename mortar_data_tag::type& mortar_data,
      const Mesh<volume_dim>& volume_mesh, const Element<volume_dim>& element,
      const typename Tags::MortarMesh<volume_dim>::type& mortar_meshes,
      const typename Tags::MortarInfo<volume_dim>::type& mortar_infos,
      const ::dg::Formulation dg_formulation,
      const DirectionMap<
          volume_dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
          face_normal_covector_and_magnitude,
      const TimeStepperType& time_stepper,
      const evolution::BoundaryCorrection& boundary_correction,
      const TimeDelta& time_step,
      const Scalar<DataVector>& gts_det_inv_jacobian,
      const VolumeArgs&... volume_args) {
    apply_impl(vars_to_update, mortar_data, volume_mesh, element, mortar_meshes,
               mortar_infos, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, time_step,
               std::numeric_limits<double>::signaling_NaN(),
               gts_det_inv_jacobian, volume_args...);
  }

  template <typename... VolumeArgs>
  static void apply(
      const gsl::not_null<typename tag_to_update::type*> vars_to_update,
      const typename mortar_data_tag::type& mortar_data,
      const Mesh<volume_dim>& volume_mesh, const Element<volume_dim>& element,
      const typename Tags::MortarMesh<volume_dim>::type& mortar_meshes,
      const typename Tags::MortarInfo<volume_dim>::type& mortar_infos,
      const ::dg::Formulation dg_formulation,
      const DirectionMap<
          volume_dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
          face_normal_covector_and_magnitude,
      const TimeStepperType& time_stepper,
      const evolution::BoundaryCorrection& boundary_correction,
      const TimeDelta& time_step, const VolumeArgs&... volume_args) {
    apply_impl(vars_to_update, mortar_data, volume_mesh, element, mortar_meshes,
               mortar_infos, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, time_step,
               std::numeric_limits<double>::signaling_NaN(), {},
               volume_args...);
  }

  // dense output (LTS only)
  template <typename... VolumeArgs>
  static void apply(
      const gsl::not_null<typename variables_tag::type*> vars_to_update,
      const typename mortar_data_tag::type& mortar_data,
      const Mesh<volume_dim>& volume_mesh, const Element<volume_dim>& element,
      const typename Tags::MortarMesh<volume_dim>::type& mortar_meshes,
      const typename Tags::MortarInfo<volume_dim>::type& mortar_infos,
      const ::dg::Formulation dg_formulation,
      const DirectionMap<
          volume_dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
          face_normal_covector_and_magnitude,
      const LtsTimeStepper& time_stepper,
      const evolution::BoundaryCorrection& boundary_correction,
      const double dense_output_time, const VolumeArgs&... volume_args) {
    apply_impl(vars_to_update, mortar_data, volume_mesh, element, mortar_meshes,
               mortar_infos, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, TimeDelta{},
               dense_output_time, {}, volume_args...);
  }

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ParallelComponent>
  static bool is_ready(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*component*/) {
    return receive_boundary_data<
        Parallel::is_dg_element_collection_v<ParallelComponent>, Metavariables,
        local_time_stepping, DenseOutput>(box, inboxes);
  }

 private:
  template <typename... VolumeArgs>
  static void apply_impl(
      const gsl::not_null<typename tag_to_update::type*> vars_to_update,
      const typename mortar_data_tag::type& mortar_data,
      const Mesh<volume_dim>& volume_mesh, const Element<volume_dim>& element,
      const typename Tags::MortarMesh<volume_dim>::type& mortar_meshes,
      const typename Tags::MortarInfo<volume_dim>::type& mortar_infos,
      const ::dg::Formulation dg_formulation,
      const DirectionMap<
          volume_dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
          face_normal_covector_and_magnitude,
      const TimeStepperType& time_stepper,
      const evolution::BoundaryCorrection& boundary_correction,
      const TimeDelta& time_step, const double dense_output_time,
      const Scalar<DataVector>& gts_det_inv_jacobian,
      const VolumeArgs&... volume_args) {
    // We treat this as a set, but use a map because we don't have a
    // non-allocating set type.
    DirectionalIdMap<volume_dim, bool> mortars_to_act_on{};
    for (const auto& [mortar, info] : mortar_infos) {
      const auto& time_stepping_policy = info.time_stepping_policy();
      switch (time_stepping_policy) {
        case TimeSteppingPolicy::EqualRate:
          if (not local_time_stepping) {
            mortars_to_act_on.emplace(mortar, true);
          }
          break;
        case TimeSteppingPolicy::Conservative:
          if (local_time_stepping) {
            mortars_to_act_on.emplace(mortar, true);
          }
          break;
        default:
          ERROR("Unhandled TimeSteppingPolicy: " << time_stepping_policy);
      }
    }
    if (mortars_to_act_on.empty()) {
      return;
    }

    tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
        detail::TemporaryReference, volume_tags_for_dg_boundary_terms>>
        volume_args_tuple{volume_args...};

    // Set up helper lambda that will compute and lift the boundary corrections
    ASSERT(
        volume_mesh.quadrature() ==
                make_array<volume_dim>(volume_mesh.quadrature(0)) or
            element.topologies() != domain::topologies::hypercube<volume_dim>,
        "Must have isotropic quadrature, but got volume mesh: " << volume_mesh);
    Scalar<DataVector> volume_det_inv_jacobian{};
    Scalar<DataVector> volume_det_jacobian{};
    if constexpr (not local_time_stepping) {
      // Need volume Jacobian for any face whose normal direction uses Gauss
      // points (i.e. not GaussLobatto or GaussRadauUpper). This means
      // mixed-quadrature non-hypercube elements (e.g. full_cylinder) where
      // some directions have collocated face points and others do not.
      const bool any_direction_uses_gauss = alg::any_of(
          volume_mesh.quadrature(), [](const Spectral::Quadrature q) {
            return q == Spectral::Quadrature::Gauss;
          });
      if (any_direction_uses_gauss) {
        get(volume_det_inv_jacobian)
            .set_data_ref(make_not_null(
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                &const_cast<DataVector&>(get(gts_det_inv_jacobian))));
        get(volume_det_jacobian) = 1.0 / get(volume_det_inv_jacobian);
      }
    }

    static_assert(
        tmpl::all<derived_boundary_corrections, std::is_final<tmpl::_1>>::value,
        "All createable classes for boundary corrections must be marked "
        "final.");
    call_with_dynamic_type<void, derived_boundary_corrections>(
        &boundary_correction,
        [&dense_output_time, &dg_formulation, &element,
         &face_normal_covector_and_magnitude, &mortar_data, &mortar_meshes,
         &mortar_infos, &mortars_to_act_on, &time_step, &time_stepper,
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

          for (const auto& mortar_id_and_data : mortar_data) {
            const auto& mortar_id = mortar_id_and_data.first;
            if (not mortars_to_act_on.contains(mortar_id)) {
              continue;
            }
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
            if (volume_mesh.basis(direction.dimension()) ==
                    Spectral::Basis::ZernikeB2 and
                volume_mesh.quadrature(direction.dimension()) ==
                    Spectral::Quadrature::GaussRadauUpper and
                direction.side() != Side::Upper) {
              ERROR(
                  "Trying to use ZernikeB2 basis with GaussRadauUpper "
                  "quadrature on the lower side: there is not a boundary here. "
                  "volume mesh: "
                  << volume_mesh << ", element ID " << element.id());
            }

            const Mesh<volume_dim - 1> face_mesh =
                volume_mesh.slice_away(direction.dimension());

            // Whether the mesh has a collocation point on this face. True for
            // GaussLobatto (points on both faces) and GaussRadauUpper (point
            // on the upper face only). When true, lifting is done via
            // lift_flux on the slice; otherwise the full Gauss-point lifting
            // path is used.
            const bool using_points_on_face =
                volume_mesh.quadrature(direction.dimension()) ==
                    Spectral::Quadrature::GaussLobatto or
                volume_mesh.quadrature(direction.dimension()) ==
                    Spectral::Quadrature::GaussRadauUpper;

            const auto compute_correction_coupling =
                [&typed_boundary_correction, &direction, dg_formulation,
                 &dt_boundary_correction_on_mortar, &face_det_jacobian,
                 &face_mesh, &face_normal_covector_and_magnitude,
                 &local_data_on_mortar, &mortar_id, &mortar_meshes,
                 &mortar_infos, &neighbor_data_on_mortar, using_points_on_face,
                 &volume_args_tuple, &volume_det_jacobian,
                 &volume_det_inv_jacobian, &volume_dt_correction, &volume_mesh](
                    const MortarData<volume_dim>& local_mortar_data,
                    const MortarData<volume_dim>& neighbor_mortar_data)
                -> DtVariables {
              if (local_time_stepping and not using_points_on_face) {
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
              ASSERT(*local_mortar_data.mortar_mesh ==
                             *neighbor_mortar_data.mortar_mesh and
                         *local_mortar_data.mortar_mesh == mortar_mesh,
                     "local mortar mesh: " << *local_mortar_data.mortar_mesh
                                           << "\nneighbor mortar mesh: "
                                           << *neighbor_mortar_data.mortar_mesh
                                           << "\nmortar mesh: " << mortar_mesh
                                           << "\n");
              const DataVector& local_data = *local_mortar_data.mortar_data;
              const DataVector& neighbor_data =
                  *neighbor_mortar_data.mortar_data;
              ASSERT(local_data.size() == neighbor_data.size(),
                     "local data size: "
                         << local_data.size()
                         << "\nneighbor_data: " << neighbor_data.size()
                         << "\n mortar_mesh: " << mortar_mesh << "\n");
              ASSERT(local_data_on_mortar.number_of_grid_points() ==
                         neighbor_data_on_mortar.number_of_grid_points(),
                     "Local data size = "
                         << local_data_on_mortar.number_of_grid_points()
                         << ", but neighbor size = "
                         << neighbor_data_on_mortar.number_of_grid_points());
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

              const std::array<Spectral::SegmentSize, volume_dim - 1>&
                  mortar_size = mortar_infos.at(mortar_id).mortar_size();

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

              if (using_points_on_face) {
                // The lift_flux function lifts only on the slice, it does not
                // add the contribution to the volume.
                ::dg::lift_flux(make_not_null(&dt_boundary_correction),
                                volume_mesh.extents(direction.dimension()),
                                magnitude_of_face_normal,
                                volume_mesh.basis(direction.dimension()));
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
              typename variables_tag::type boundary_lifted_data{};
              auto& lifted_data =
                  using_points_on_face ? boundary_lifted_data : *vars_to_update;
              if (using_points_on_face) {
                lifted_data.initialize(face_mesh.number_of_grid_points(), 0.0);
              }

              const auto& mortar_data_history = mortar_id_and_data.second;
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

              if (using_points_on_face) {
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
              auto& lifted_data = using_points_on_face
                                      ? dt_boundary_correction_on_mortar
                                      : volume_dt_correction;
              lifted_data = compute_correction_coupling(
                  mortar_id_and_data.second.local(),
                  mortar_id_and_data.second.neighbor());

              if (using_points_on_face) {
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
namespace ApplyBoundaryCorrections_detail {
template <bool LocalTimeStepping, size_t VolumeDim, bool DenseOutput,
          bool UseNodegroupDgElements>
struct ActionImpl {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          VolumeDim, UseNodegroupDgElements>>;
  using const_global_cache_tags =
      tmpl::list<evolution::Tags::BoundaryCorrection, ::dg::Tags::Formulation>;

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
        "The action is told by the template parameter UseNodegroupDgElements "
        "that it is being used with a DgElementCollection, but the "
        "ParallelComponent is not a DgElementCollection. You need to change "
        "the template parameter on the action in your action list.");
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    const Element<volume_dim>& element =
        db::get<domain::Tags::Element<volume_dim>>(box);

    if (UNLIKELY(element.number_of_neighbors() == 0)) {
      // We have no neighbors, yay!
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    if (not receive_boundary_data<
            Parallel::is_dg_element_collection_v<ParallelComponent>,
            Metavariables, LocalTimeStepping, false>(make_not_null(&box),
                                                     make_not_null(&inboxes))) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    // LTS updates the evolved variables, so we can skip that if they
    // are unused.  GTS updates the derivatives, which are always
    // needed to update the history.
    if (LocalTimeStepping and
        ::SelfStart::step_unused(
            db::get<::Tags::TimeStepId>(box),
            db::get<::Tags::Next<::Tags::TimeStepId>>(box))) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    db::mutate_apply<ApplyBoundaryCorrections<LocalTimeStepping, Metavariables,
                                              VolumeDim, DenseOutput>>(
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace ApplyBoundaryCorrections_detail

/*!
 * \brief Computes the boundary corrections for global time-stepping
 * and adds them to the time derivative.
 */
template <size_t VolumeDim, bool UseNodegroupDgElements>
struct ApplyBoundaryCorrectionsToTimeDerivative
    : ApplyBoundaryCorrections_detail::ActionImpl<false, VolumeDim, false,
                                                  UseNodegroupDgElements> {};

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
template <size_t VolumeDim, bool DenseOutput, bool UseNodegroupDgElements>
struct ApplyLtsBoundaryCorrections
    : ApplyBoundaryCorrections_detail::ActionImpl<true, VolumeDim, DenseOutput,
                                                  UseNodegroupDgElements> {};
}  // namespace Actions
}  // namespace evolution::dg
