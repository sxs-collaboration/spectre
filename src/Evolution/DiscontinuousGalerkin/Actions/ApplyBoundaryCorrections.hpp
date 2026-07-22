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
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
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
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
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
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/TMPL.hpp"

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

template <typename BoundaryCorrectionClass>
struct get_dg_auxiliary_boundary_terms {
  using type = evolution::dg::Actions::detail::
      get_dg_auxiliary_boundary_terms_volume_tags_or_default_t<
          BoundaryCorrectionClass, tmpl::list<>>;
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
///
/// If \p LocalTimeStepping is true, it will process all data
/// necessary for conservative LTS, otherwise it will process all data
/// necessary for GTS.  Some data for the other mode may also be
/// processed to simplify the message handling.
template <bool UseNodegroupDgElements, typename Metavariables,
          bool LocalTimeStepping, bool DenseOutput,
          bool ComputeAuxiliary = false, typename DbTagsList,
          typename... InboxTags>
bool receive_boundary_data(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes) {
  constexpr size_t volume_dim = Metavariables::system::volume_dim;
  constexpr size_t face_dim = volume_dim - 1;
  static_assert(LocalTimeStepping or not DenseOutput,
                "Should not be receiving data for dense output with GTS.");

  auto& inbox =
      tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          volume_dim, UseNodegroupDgElements, ComputeAuxiliary>>(*inboxes);

  const auto& volume_mesh = db::get<domain::Tags::Mesh<volume_dim>>(*box);
  const auto& mortar_infos = db::get<Tags::MortarInfo<volume_dim>>(*box);
  const auto& mortar_next_time_step_ids =
      db::get<evolution::dg::Tags::MortarNextTemporalId<volume_dim>>(*box);

  for (;;) {
    std::optional<TimeStepId> time_to_process{};
    for (const auto& [mortar_id, mortar_next_time_step_id] :
         mortar_next_time_step_ids) {
      if (time_to_process.has_value() and
          mortar_next_time_step_id > *time_to_process) {
        continue;
      }

      const auto& time_stepping_policy =
          mortar_infos.at(mortar_id).time_stepping_policy();
      switch (time_stepping_policy) {
        case TimeSteppingPolicy::EqualRate:
          if (LocalTimeStepping or
              mortar_next_time_step_id > db::get<::Tags::TimeStepId>(*box)) {
            continue;
          }
          break;
        case TimeSteppingPolicy::Conservative:
          if constexpr (not LocalTimeStepping) {
            continue;
          } else {
            const LtsTimeStepper& time_stepper =
                db::get<::Tags::TimeStepper<LtsTimeStepper>>(*box);
            using goal_tag =
                tmpl::conditional_t<DenseOutput, ::Tags::Time,
                                    ::Tags::Next<::Tags::TimeStepId>>;
            const auto& goal = db::get<goal_tag>(*box);
            if (not time_stepper.neighbor_data_required(
                    goal, mortar_next_time_step_id)) {
              continue;
            }
          }
          break;
        default:
          ERROR("Unhandled TimeSteppingPolicy: " << time_stepping_policy);
      }

      time_to_process.emplace(mortar_next_time_step_id);
    }

    if (not time_to_process.has_value()) {
      if constexpr (using_subcell_v<Metavariables> and not LocalTimeStepping) {
        evolution::dg::subcell::neighbor_reconstructed_face_solution<
            volume_dim, typename Metavariables::SubcellOptions::
                            DgComputeSubcellNeighborPackagedData>(
            &db::as_access(*box));
      }
      return true;
    }

    const auto& element = db::get<domain::Tags::Element<volume_dim>>(*box);
    const auto expected_messages = static_cast<size_t>(alg::accumulate(
        mortar_next_time_step_ids, 0,
        [&mortar_infos, &element, &time_to_process](const size_t total,
                                                    const auto& entry) {
          if (entry.second != *time_to_process) {
            return total;
          } else if (mortar_infos.at(entry.first).interface_data_policy() !=
                     InterfaceDataPolicy::NonconformingNeighborInterpolates) {
            return total + 1;
          } else {
            return total +
                   element.neighbors().at(entry.first.direction()).size();
          }
        }));

    // This is a
    //
    // std::map<TimeStepId,
    //          V<std::pair<DirectionalId<volume_dim>,
    //                      evolution::dg::BoundaryData<volume_dim>>>,
    //
    // where V<> is a vector-like type the details of which we don't
    // want to hardcode here.
    auto& inbox_data = inbox.messages;
    auto messages_to_process = inbox_data.end();

    {
      size_t missing_messages{};
      do {
        inbox.collect_messages();
        if (messages_to_process == inbox_data.end()) {
          messages_to_process = inbox_data.find(*time_to_process);
        }
        const size_t available_messages =
            messages_to_process == inbox_data.end()
                ? 0
                : messages_to_process->second.size();
        ASSERT(available_messages <= expected_messages,
               "Too many boundary messages at " << *time_to_process << ": "
                                                << available_messages << "/"
                                                << expected_messages);
        missing_messages = expected_messages - available_messages;
      } while (missing_messages != 0 and
               inbox.set_missing_messages(missing_messages));
      if (missing_messages != 0) {
        return false;
      }
    }

    // *time_to_process represents the same temporal event as this,
    // but may have an out-of-date slab size because the
    // MortarNextTemporalId data can be sent before the slab size is
    // chosen.  It is important that the corrected version be what is
    // inserted into the boundary history.
    const TimeStepId processing_time = messages_to_process->first;
    std::unordered_map<Direction<volume_dim>, std::vector<size_t>>
        contributors_multiple_non_conforming_neighbors{};

    for (auto& mortar_id_and_data : messages_to_process->second) {
      const auto& received_mortar_id = mortar_id_and_data.first;
      auto& received_mortar_data = mortar_id_and_data.second;
      const auto& direction = received_mortar_id.direction();
      const auto& neighbor_mesh = received_mortar_data.volume_mesh;
      const size_t sliced_away_dim = direction.dimension();
      const Mesh<face_dim> face_mesh = volume_mesh.slice_away(sliced_away_dim);
      // If there are multiple non-conforming neighbors, there is only a
      // single mortar labeled by the host ElementId.  This is done
      // because the data from all neighbors will be combined onto a
      // single mortar as it makes no sense to have multiple mortars
      // between non-conforming Elements.
      const DirectionalId<volume_dim> mortar_id =
          mortar_infos.contains(received_mortar_id)
              ? received_mortar_id
              : DirectionalId<volume_dim>{direction, element.id()};

      ASSERT(mortar_next_time_step_ids.at(mortar_id) == processing_time or
                 contributors_multiple_non_conforming_neighbors.contains(
                     direction),
             "Processing wrong time for mortar "
                 << mortar_id << "\nExpected "
                 << mortar_next_time_step_ids.at(mortar_id)
                 << " but processing " << processing_time);

      const auto& time_stepping_policy =
          mortar_infos.at(mortar_id).time_stepping_policy();

      if constexpr (using_subcell_v<Metavariables>) {
        if (time_stepping_policy == TimeSteppingPolicy::EqualRate) {
          evolution::dg::subcell::receive_subcell_data_for_dg<volume_dim>(
              &db::as_access(*box), mortar_id, received_mortar_data);
          evolution::dg::subcell::neighbor_tci_decision<volume_dim>(
              make_not_null(&db::as_access(*box)), mortar_id,
              received_mortar_data);
        }
      }

      db::mutate<evolution::dg::Tags::MortarMesh<volume_dim>,
                 evolution::dg::Tags::MortarData<volume_dim>,
                 evolution::dg::Tags::MortarDataHistory<volume_dim>,
                 evolution::dg::Tags::MortarNextTemporalId<volume_dim>,
                 domain::Tags::NeighborMesh<volume_dim>>(
          [&](const gsl::not_null<DirectionalIdMap<volume_dim, Mesh<face_dim>>*>
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
              [[maybe_unused]] const gsl::not_null<
                  DirectionalIdMap<volume_dim, TimeStepId>*>
                  mortar_next_time_step_ids_mutable,
              const gsl::not_null<
                  DirectionalIdMap<volume_dim, Mesh<volume_dim>>*>
                  neighbor_meshes) {
            switch (mortar_infos.at(mortar_id).interface_data_policy()) {
              case InterfaceDataPolicy::CopyProject:
                [[fallthrough]];
              case InterfaceDataPolicy::OrientCopyProject: {
                neighbor_meshes->insert_or_assign(received_mortar_id,
                                                  neighbor_mesh);
                const Mesh<face_dim> neighbor_face_mesh =
                    received_mortar_data.volume_mesh.slice_away(
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

                // The auxiliary pass must not advance the mortar clock: the
                // physical receive later in the same step still reads it.
                if constexpr (not ComputeAuxiliary) {
                  mortar_next_time_step_ids_mutable->at(mortar_id) =
                      received_mortar_data.validity_range;
                }

                ASSERT(using_subcell_v<Metavariables> or
                           received_mortar_data.boundary_correction_data
                               .has_value(),
                       "Must receive neighbor boundary correction data when "
                       "not using DG-subcell. Mortar ID is: ("
                           << mortar_id.direction() << "," << mortar_id.id()
                           << ") and TimeStepId is " << processing_time);
                MortarData<volume_dim> neighbor_mortar_data{};
                neighbor_mortar_data.face_mesh = neighbor_face_mesh;
                neighbor_mortar_data.mortar_mesh =
                    received_mortar_data.boundary_correction_mesh;
                neighbor_mortar_data.mortar_data =
                    std::move(received_mortar_data.boundary_correction_data);
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
                        processing_time, received_mortar_data.integration_order,
                        std::move(neighbor_mortar_data));
                    boundary_data_history->at(mortar_id).remote().for_each(
                        project_boundary_mortar_data);
                    break;
                  default:
                    ERROR("Unhandled TimeSteppingPolicy: "
                          << time_stepping_policy);
                }
                break;
              }
              case InterfaceDataPolicy::NonconformingSelfInterpolates: {
                if constexpr (volume_dim > 1) {
                  neighbor_meshes->insert_or_assign(received_mortar_id,
                                                    neighbor_mesh);
                  if constexpr (not ComputeAuxiliary) {
                    mortar_next_time_step_ids_mutable->at(mortar_id) =
                        received_mortar_data.validity_range;
                  }
                  mortar_meshes->at(mortar_id) = face_mesh;
                  gts_mortar_data->at(mortar_id).neighbor().face_mesh =
                      face_mesh;
                  gts_mortar_data->at(mortar_id).neighbor().mortar_mesh =
                      face_mesh;
                  const auto& interpolator =
                      mortar_infos.at(mortar_id).interpolator().value();
                  const auto& received_data =
                      received_mortar_data.boundary_correction_data.value();
                  DataVector interpolated_data =
                      interpolator.interpolate_to_host(received_data);
                  gts_mortar_data->at(mortar_id).neighbor().mortar_data =
                      std::move(interpolated_data);
                } else {
                  ERROR("Cannot have non-conforming neighbors in 1D");
                }
                break;
              }
              case InterfaceDataPolicy::NonconformingNeighborInterpolates: {
                if constexpr (volume_dim > 1) {
                  // We do not insert the neighbor mesh into neighbor_meshes
                  // as this could overflow the FixedHashMap size
                  const size_t npts_mortar = face_mesh.number_of_grid_points();
                  const size_t mortar_data_size = gts_mortar_data->at(mortar_id)
                                                      .local()
                                                      .mortar_data.value()
                                                      .size();
                  const size_t number_of_components =
                      mortar_data_size / npts_mortar;
                  // The data received from each neighbor has been interpolated
                  // to a subset of points of the single mortar mesh of the host
                  // If this is the first neighbor processed,
                  if (not contributors_multiple_non_conforming_neighbors
                              .contains(direction)) {
                    contributors_multiple_non_conforming_neighbors.emplace(
                        direction, std::vector<size_t>(
                                       face_mesh.number_of_grid_points(), 0));
                    if constexpr (not ComputeAuxiliary) {
                      mortar_next_time_step_ids_mutable->at(mortar_id) =
                          received_mortar_data.validity_range;
                    }
                    mortar_meshes->at(mortar_id) = face_mesh;
                    gts_mortar_data->at(mortar_id).neighbor().face_mesh =
                        face_mesh;
                    gts_mortar_data->at(mortar_id).neighbor().mortar_mesh =
                        face_mesh;
                    gts_mortar_data->at(mortar_id).neighbor().mortar_data =
                        DataVector{mortar_data_size, 0.0};
                  }
                  if constexpr (not ComputeAuxiliary) {
                    ASSERT(mortar_next_time_step_ids_mutable->at(mortar_id) ==
                               received_mortar_data.validity_range,
                           "Inconsistent validity range "
                               << received_mortar_data.validity_range
                               << " received from " << received_mortar_id
                               << "; expected "
                               << mortar_next_time_step_ids_mutable->at(
                                      mortar_id));
                  }
                  const auto& interpolated_boundary_data =
                      received_mortar_data.interpolated_boundary_data.value();
                  const auto& interpolated_data =
                      interpolated_boundary_data.boundary_data();
                  const size_t interpolated_data_size =
                      interpolated_data.size();
                  const auto& offsets = interpolated_boundary_data.offsets();
                  const size_t npts_interpolated = offsets.size();
                  ASSERT(npts_interpolated * number_of_components ==
                             interpolated_data_size,
                         "Size mismatch!  Number of interpolated points "
                             << npts_interpolated
                             << " times number of components "
                             << number_of_components
                             << " is not interpolated data size "
                             << interpolated_data_size);
                  auto& target_mortar_data = gts_mortar_data->at(mortar_id)
                                                 .neighbor()
                                                 .mortar_data.value();
                  auto& contributors =
                      contributors_multiple_non_conforming_neighbors.at(
                          direction);
                  for (size_t i = 0; i < npts_interpolated; ++i) {
                    ++contributors[offsets[i]];
                    for (size_t c = 0; c < number_of_components; ++c) {
                      target_mortar_data[offsets[i] + c * npts_mortar] +=
                          interpolated_data[i + c * npts_interpolated];
                    }
                  }
                } else {
                  ERROR("Cannot have non-conforming neighbors in 1D");
                }
                break;
              }
              default:
                ERROR("InterfaceDataPolicy "
                      << mortar_infos.at(mortar_id).interface_data_policy()
                      << " is not handled yet, id = " << mortar_id);
            }
          },
          box);
    }

    db::mutate<evolution::dg::Tags::MortarData<volume_dim>>(
        [&contributors_multiple_non_conforming_neighbors, &element](
            const gsl::not_null<DirectionalIdMap<
                volume_dim, evolution::dg::MortarDataHolder<volume_dim>>*>
                gts_mortar_data) {
          for (const auto& [direction, contributors] :
               contributors_multiple_non_conforming_neighbors) {
            const DirectionalId<volume_dim> mortar_id =
                DirectionalId<volume_dim>{direction, element.id()};
            auto& target_mortar_data =
                gts_mortar_data->at(mortar_id).neighbor().mortar_data.value();
            const size_t npts_mortar = contributors.size();
            const size_t number_of_components =
                target_mortar_data.size() / npts_mortar;
            ASSERT(alg::none_of(contributors,
                                [](const size_t n) { return n == 0; }),
                   "Not all points were interpolated.  Direction = "
                       << direction << " ElementId = " << element.id() << "\n"
                       << "target_mortar_data = " << target_mortar_data);
            for (size_t i = 0; i < npts_mortar; ++i) {
              for (size_t c = 0; c < number_of_components; ++c) {
                target_mortar_data[i + c * npts_mortar] /=
                    static_cast<double>(contributors[i]);
              }
            }
          }
        },
        box);

    inbox_data.erase(messages_to_process);
    if constexpr (ComputeAuxiliary) {
      // The auxiliary pass processes a single temporal id; return rather than
      // looping for more ready times.
      return true;
    }
  }
}

/// Apply corrections from boundary communication.
///
/// This is usually used indirectly through
/// `ApplyBoundaryCorrectionsToTimeDerivative`,
/// `ApplyLtsBoundaryCorrections`, or `ApplyLtsDenseBoundaryCorrections`.
///
/// If `LocalTimeStepping` is false, updates the derivative of the variables,
/// which should be done before taking a time step.  If
/// `LocalTimeStepping` is true, updates the variables themselves, which should
/// be done after the volume update.
///
/// Setting \p DenseOutput to true receives data required for output
/// at ::Tags::Time instead of performing a full step.  This is only
/// used for local time-stepping.
template <bool LocalTimeStepping, typename Metavariables, bool DenseOutput,
          bool ComputeAuxiliary = false>
struct ApplyBoundaryCorrections {
  static constexpr bool local_time_stepping = LocalTimeStepping;
  static_assert(local_time_stepping or not DenseOutput,
                "GTS does not use ApplyBoundaryCorrections for dense output.");
  static_assert(not(ComputeAuxiliary and local_time_stepping),
                "Auxiliary boundary corrections are not supported with LTS.");

  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = system::volume_dim;
  using variables_tag = typename system::variables_tag;
  using FilterTagList = typename variables_tag::tags_list;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using auxiliary_variables_tag = ::Tags::Variables<
      evolution::dg::Actions::detail::get_auxiliary_variables_or_default_t<
          system, tmpl::list<>>>;
  // The correction-buffer type. The physical/LTS paths hold time derivatives
  // (`dt_variables_tag`), lifted into `dt_variables_tag` (GTS) or
  // `variables_tag` (LTS, via the prefix-agnostic `Variables` arithmetic). The
  // auxiliary variables are not time-evolved, so the auxiliary pass's buffer is
  // shaped by `auxiliary_variables` directly (no `dt` prefix); it holds the
  // correction added to the auxiliary variables.
  using DtVariables =
      tmpl::conditional_t<ComputeAuxiliary,
                          typename auxiliary_variables_tag::type,
                          typename dt_variables_tag::type>;
  using derived_boundary_corrections =
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               evolution::BoundaryCorrection>;

  using volume_tags_for_dg_boundary_terms =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
          derived_boundary_corrections,
          tmpl::conditional_t<ComputeAuxiliary,
                              detail::get_dg_auxiliary_boundary_terms<tmpl::_1>,
                              detail::get_dg_boundary_terms<tmpl::_1>>>>>;

  using TimeStepperType =
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>;

  // The auxiliary pass writes the corrected auxiliary variables into their own
  // storage (`auxiliary_variables_tag`); LTS updates `variables_tag`; the GTS
  // physical pass updates `dt_variables_tag`.
  using tag_to_update =
      tmpl::conditional_t<ComputeAuxiliary, auxiliary_variables_tag,
                          tmpl::conditional_t<local_time_stepping,
                                              variables_tag, dt_variables_tag>>;
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
          tmpl::conditional_t<
              DenseOutput, ::Tags::Time,
              tmpl::list<
                  ::Tags::TimeStep,
                  Filters::Tags::SpectralFilter<volume_dim, FilterTagList>,
                  ::Tags::StepNumberWithinSlab,
                  domain::Tags::Jacobian<volume_dim, Frame::Grid,
                                         Frame::Inertial>,
                  domain::Tags::InverseJacobian<volume_dim, Frame::Grid,
                                                Frame::Inertial>>>,
          tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                              domain::Tags::DetInvJacobian<
                                  Frame::ElementLogical, Frame::Inertial>>>>,
      volume_tags_for_dg_boundary_terms>;

  // full step (GTS: local_time_stepping=false, DenseOutput=false)
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
      const Filters::Filter<volume_dim, FilterTagList>& boundary_filter,
      const uint64_t step_number_within_slab,
      const Jacobian<DataVector, volume_dim, Frame::Grid, Frame::Inertial>&
          volume_jac_grid_to_inertial,
      const InverseJacobian<DataVector, volume_dim, Frame::Grid,
                            Frame::Inertial>& volume_inv_jac_grid_to_inertial,
      const Scalar<DataVector>& gts_det_inv_jacobian,
      const VolumeArgs&... volume_args) {
    apply_impl(vars_to_update, mortar_data, volume_mesh, element, mortar_meshes,
               mortar_infos, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, time_step,
               std::numeric_limits<double>::signaling_NaN(), &boundary_filter,
               step_number_within_slab, volume_jac_grid_to_inertial,
               volume_inv_jac_grid_to_inertial, gts_det_inv_jacobian,
               volume_args...);
  }

  // full step (LTS: local_time_stepping=true, DenseOutput=false)
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
      const Filters::Filter<volume_dim, FilterTagList>& boundary_filter,
      const uint64_t step_number_within_slab,
      const Jacobian<DataVector, volume_dim, Frame::Grid, Frame::Inertial>&
          volume_jac_grid_to_inertial,
      const InverseJacobian<DataVector, volume_dim, Frame::Grid,
                            Frame::Inertial>& volume_inv_jac_grid_to_inertial,
      const VolumeArgs&... volume_args) {
    apply_impl(vars_to_update, mortar_data, volume_mesh, element, mortar_meshes,
               mortar_infos, dg_formulation, face_normal_covector_and_magnitude,
               time_stepper, boundary_correction, time_step,
               std::numeric_limits<double>::signaling_NaN(), &boundary_filter,
               step_number_within_slab, volume_jac_grid_to_inertial,
               volume_inv_jac_grid_to_inertial, {}, volume_args...);
  }

  // dense output (LTS only, DenseOutput=true)
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
    apply_impl(
        vars_to_update, mortar_data, volume_mesh, element, mortar_meshes,
        mortar_infos, dg_formulation, face_normal_covector_and_magnitude,
        time_stepper, boundary_correction, TimeDelta{}, dense_output_time,
        nullptr, static_cast<uint64_t>(0),
        Jacobian<DataVector, volume_dim, Frame::Grid, Frame::Inertial>{},
        InverseJacobian<DataVector, volume_dim, Frame::Grid, Frame::Inertial>{},
        {}, volume_args...);
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
        local_time_stepping, DenseOutput, ComputeAuxiliary>(box, inboxes);
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
      const Filters::Filter<volume_dim, FilterTagList>* const filter_ptr,
      const uint64_t step_number_within_slab,
      const Jacobian<DataVector, volume_dim, Frame::Grid, Frame::Inertial>&
          volume_jac_grid_to_inertial,
      const InverseJacobian<DataVector, volume_dim, Frame::Grid,
                            Frame::Inertial>& volume_inv_jac_grid_to_inertial,
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

    bool boundary_filter_active = false;
    // The auxiliary boundary correction is not filtered currently (see the
    // filter application below), so the auxiliary pass skips the
    // filter-activity check and the face-Jacobian setup it would trigger.
    if constexpr (not DenseOutput and not ComputeAuxiliary) {
      if (filter_ptr != nullptr and
          dynamic_cast<const Filters::None<volume_dim, FilterTagList>*>(
              filter_ptr) == nullptr) {
        const auto step_number = static_cast<size_t>(step_number_within_slab);
        boundary_filter_active =
            filter_ptr->apply_boundary_filter_on_substep() or
            filter_ptr->apply_boundary_filter_on_this_step(step_number);
      }
    }

    const bool need_face_jacobians = [&]() {
      if constexpr (DenseOutput) {
        return false;
      } else {
        return boundary_filter_active and filter_ptr != nullptr and
               filter_ptr->need_jacobians();
      }
    }();

    size_t max_face_grid_points = 0;
    if (need_face_jacobians) {
      for (const auto& [mortar_id, _info] : mortar_infos) {
        if (not mortars_to_act_on.contains(mortar_id) or
            mortar_id.id() == ElementId<volume_dim>::external_boundary_id()) {
          continue;
        }
        max_face_grid_points =
            std::max(max_face_grid_points,
                     volume_mesh.slice_away(mortar_id.direction().dimension())
                         .number_of_grid_points());
      }
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::unique_ptr<double[]> face_jac_buffer{nullptr};
    if (max_face_grid_points > 0) {
      constexpr size_t jac_components =
          Jacobian<DataVector, volume_dim, Frame::Grid,
                   Frame::Inertial>::size();
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      face_jac_buffer = cpp20::make_unique_for_overwrite<double[]>(
          2 * jac_components * max_face_grid_points);
    }

    std::optional<
        Jacobian<DataVector, volume_dim, Frame::Grid, Frame::Inertial>>
        face_jac_grid_to_inertial{};
    std::optional<
        InverseJacobian<DataVector, volume_dim, Frame::Grid, Frame::Inertial>>
        face_inv_jac_grid_to_inertial{};
    if (face_jac_buffer != nullptr) {
      face_jac_grid_to_inertial.emplace();
      face_inv_jac_grid_to_inertial.emplace();
    }
    std::optional<Direction<volume_dim>> cached_face_jac_direction{};

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
        [&cached_face_jac_direction, &dense_output_time, &dg_formulation,
         &element, &face_inv_jac_grid_to_inertial,
         &face_jac_buffer,  // NOLINT(modernize-avoid-c-arrays)
         &face_jac_grid_to_inertial, &face_normal_covector_and_magnitude,
         boundary_filter_active, filter_ptr, max_face_grid_points,
         need_face_jacobians, &mortar_data, &mortar_meshes, &mortar_infos,
         &mortars_to_act_on, &time_step, &time_stepper, &vars_to_update,
         &volume_args_tuple, &volume_det_jacobian, &volume_det_inv_jacobian,
         &volume_inv_jac_grid_to_inertial, &volume_jac_grid_to_inertial,
         &volume_mesh](auto* typed_boundary_correction) {
          (void)need_face_jacobians;
          using BcType = std::decay_t<decltype(*typed_boundary_correction)>;
          // Compute internal boundary quantities on the mortar for sides of
          // the element that have neighbors, i.e. they are not an external
          // side.  The auxiliary arm uses the detect-or-default helper so it
          // remains well-formed for any correction, even those lacking the
          // alias; this keeps the change additive.
          using mortar_tags_list = tmpl::conditional_t<
              ComputeAuxiliary,
              evolution::dg::Actions::detail::
                  get_dg_auxiliary_package_field_tags_or_default_t<
                      BcType, tmpl::list<>>,
              typename BcType::dg_package_field_tags>;

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

            if (need_face_jacobians and
                (not cached_face_jac_direction.has_value() or
                 *cached_face_jac_direction != direction)) {
              constexpr size_t jac_components =
                  Jacobian<DataVector, volume_dim, Frame::Grid,
                           Frame::Inertial>::size();
              const size_t current_face_size =
                  face_mesh.number_of_grid_points();
              for (size_t i = 0; i < jac_components; ++i) {
                (*face_jac_grid_to_inertial)[i].set_data_ref(
                    &face_jac_buffer[i * max_face_grid_points],
                    current_face_size);
                (*face_inv_jac_grid_to_inertial)[i].set_data_ref(
                    &face_jac_buffer[(jac_components + i) *
                                     max_face_grid_points],
                    current_face_size);
              }
              ::dg::project_tensor_to_boundary(
                  make_not_null(&*face_jac_grid_to_inertial),
                  volume_jac_grid_to_inertial, volume_mesh, direction);
              ::dg::project_tensor_to_boundary(
                  make_not_null(&*face_inv_jac_grid_to_inertial),
                  volume_inv_jac_grid_to_inertial, volume_mesh, direction);
              cached_face_jac_direction = direction;
            }

            const auto compute_correction_coupling =
                [&typed_boundary_correction, boundary_filter_active, &direction,
                 dg_formulation, &dt_boundary_correction_on_mortar,
                 &face_det_jacobian, &face_inv_jac_grid_to_inertial,
                 &face_jac_grid_to_inertial, &face_mesh,
                 &face_normal_covector_and_magnitude, filter_ptr,
                 &local_data_on_mortar, &mortar_id, &mortar_meshes,
                 &mortar_infos, &neighbor_data_on_mortar, using_points_on_face,
                 &volume_args_tuple, &volume_det_jacobian,
                 &volume_det_inv_jacobian, &volume_dt_correction, &volume_mesh,
                 &element](const MortarData<volume_dim>& local_mortar_data,
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

              if constexpr (ComputeAuxiliary) {
                // `dt_boundary_correction_on_mortar` is used purely as a
                // scratch container for the packaged auxiliary correction on
                // the mortar.  The correction is ultimately applied to the
                // auxiliary-variable storage, not to a time derivative:
                // `tag_to_update` is `auxiliary_variables_tag` for the
                // auxiliary pass, and this `dt`-prefixed buffer reconciles with
                // it via the prefix-agnostic `Variables::operator+=` /
                // `add_slice_to_data` used when the lifted result is added to
                // `vars_to_update`.
                using aux_volume_tags = evolution::dg::Actions::detail::
                    get_dg_auxiliary_boundary_terms_volume_tags_or_default_t<
                        BcType, tmpl::list<>>;
                call_auxiliary_boundary_correction(
                    make_not_null(&dt_boundary_correction_on_mortar),
                    local_data_on_mortar, neighbor_data_on_mortar,
                    *typed_boundary_correction, dg_formulation,
                    volume_args_tuple, aux_volume_tags{});
              } else {
                call_boundary_correction(
                    make_not_null(&dt_boundary_correction_on_mortar),
                    local_data_on_mortar, neighbor_data_on_mortar,
                    *typed_boundary_correction, dg_formulation,
                    volume_args_tuple,
                    typename BcType::dg_boundary_terms_volume_tags{});
              }

              const std::array<Spectral::SegmentSize, volume_dim - 1>&
                  mortar_size = mortar_infos.at(mortar_id).mortar_size();

              // This cannot reuse an allocation because it is initialized
              // via move-assignment.  (If it is used at all.)
              DtVariables dt_boundary_correction_projected_onto_face{};
              auto& dt_boundary_correction =
                  [&dt_boundary_correction_on_mortar,
                   &dt_boundary_correction_projected_onto_face, &face_mesh,
                   &mortar_mesh, &mortar_size, &element,
                   &direction]() -> DtVariables& {
                if (element.neighbors().at(direction).are_conforming() and
                    Spectral::needs_projection(face_mesh, mortar_mesh,
                                               mortar_size)) {
                  dt_boundary_correction_projected_onto_face =
                      ::dg::project_from_mortar(
                          dt_boundary_correction_on_mortar, face_mesh,
                          mortar_mesh, mortar_size);
                  return dt_boundary_correction_projected_onto_face;
                }
                return dt_boundary_correction_on_mortar;
              }();
              // The auxiliary boundary correction is not filtered currently
              if constexpr (not DenseOutput and not ComputeAuxiliary) {
                // Filter the boundary correction on the mortar before it is
                // lifted into the volume.
                if (boundary_filter_active) {
                  using BoundaryFilterVars = Variables<FilterTagList>;
                  auto boundary_filter_view =
                      dt_boundary_correction
                          .template reference_with_different_prefixes<
                              BoundaryFilterVars>();
                  filter_ptr->apply_on_boundary(
                      make_not_null(&boundary_filter_view), face_mesh,
                      face_inv_jac_grid_to_inertial, face_jac_grid_to_inertial);
                }
              } else {
                (void)boundary_filter_active;
                (void)face_inv_jac_grid_to_inertial;
                (void)face_jac_grid_to_inertial;
                (void)filter_ptr;
              }

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

  template <typename... BoundaryCorrectionTags, typename... Tags,
            typename BoundaryCorrection, typename... AllVolumeArgs,
            typename... VolumeTagsForCorrection>
  static void call_auxiliary_boundary_correction(
      const gsl::not_null<Variables<tmpl::list<BoundaryCorrectionTags...>>*>
          boundary_corrections_on_mortar,
      const Variables<tmpl::list<Tags...>>& local_boundary_data,
      const Variables<tmpl::list<Tags...>>& neighbor_boundary_data,
      const BoundaryCorrection& boundary_correction,
      const ::dg::Formulation dg_formulation,
      const tuples::TaggedTuple<detail::TemporaryReference<AllVolumeArgs>...>&
          volume_args_tuple,
      tmpl::list<VolumeTagsForCorrection...> /*meta*/) {
    boundary_correction.dg_auxiliary_boundary_terms(
        make_not_null(
            &get<BoundaryCorrectionTags>(*boundary_corrections_on_mortar))...,
        get<Tags>(local_boundary_data)..., get<Tags>(neighbor_boundary_data)...,
        dg_formulation,
        tuples::get<detail::TemporaryReference<VolumeTagsForCorrection>>(
            volume_args_tuple)...);
  }
};

/// Apply corrections from boundary communication for LTS dense output.
template <typename Metavariables>
struct ApplyLtsDenseBoundaryCorrections
    : ApplyBoundaryCorrections<true, Metavariables, true> {};

namespace Actions {
namespace ApplyBoundaryCorrections_detail {
template <bool LocalTimeStepping, size_t VolumeDim, bool DenseOutput,
          bool UseNodegroupDgElements, bool ComputeAuxiliary = false>
struct ActionImpl {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          VolumeDim, UseNodegroupDgElements, ComputeAuxiliary>>;
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
    static_assert(Metavariables::system::volume_dim == VolumeDim);
    // The LDG auxiliary pass is implemented for DG elements only for now;
    // DG-subcell support is deferred.
    static_assert(
        not(ComputeAuxiliary and evolution::dg::using_subcell_v<Metavariables>),
        "LDG auxiliary boundary corrections do not support DG-subcell yet.");
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
            Metavariables, LocalTimeStepping, false, ComputeAuxiliary>(
            make_not_null(&box), make_not_null(&inboxes))) {
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
                                              DenseOutput, ComputeAuxiliary>>(
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
 * \brief Receives and lifts the LDG auxiliary boundary corrections into the
 * auxiliary variables.
 *
 * This is the "receive" counterpart of the LDG auxiliary send action. It is the
 * first communication step of the LDG two-communication scheme: after this
 * action runs, the auxiliary variables have been corrected with the numerical
 * flux. The second step (the physical boundary correction) is done by
 * `ApplyBoundaryCorrectionsToTimeDerivative`.
 */
template <size_t VolumeDim, bool UseNodegroupDgElements>
struct ApplyAuxiliaryBoundaryCorrectionsToVariables
    : ApplyBoundaryCorrections_detail::ActionImpl<false, VolumeDim, false,
                                                  UseNodegroupDgElements,
                                                  /*ComputeAuxiliary=*/true> {};

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
template <size_t VolumeDim, bool UseNodegroupDgElements>
struct ApplyLtsBoundaryCorrections
    : ApplyBoundaryCorrections_detail::ActionImpl<true, VolumeDim, false,
                                                  UseNodegroupDgElements> {};
}  // namespace Actions
}  // namespace evolution::dg
