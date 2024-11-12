// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/FlatLogicalMetric.hpp"
#include "Domain/JacobianDiagnostic.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/OptionTags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ContributeMemoryData.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "ParallelAlgorithms/Amr/Actions/Initialize.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/TruncationError.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/Events/MonitorMemory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/SlabCompares.hpp"
#include "Time/Triggers/TimeCompares.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

/// \cond
using MinGridSpacingReductionData = Parallel::ReductionData<
    // Time
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    // Minimum grid spacing
    Parallel::ReductionDatum<double, funcl::Min<>>>;

struct MinGridSpacingFormatter
    : tt::ConformsTo<observers::protocols::ReductionDataFormatter> {
  using reduction_data = MinGridSpacingReductionData;
  std::string operator()(const double time, const double min_grid_spacing) {
    return "Time: " + get_output(time) +
           ", Global inertial minimum grid spacing: " +
           get_output(min_grid_spacing);
  }
  void pup(PUP::er& /*p*/) {}
};

namespace Actions {
template <size_t Dim>
struct ExportCoordinates {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) {
    return {observers::TypeOfObservation::Volume,
            observers::ObservationKey("ObserveCoords")};
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const double time = get<Tags::Time>(box);

    const auto& mesh = get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(box);
    const auto& inertial_coordinates =
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto deriv_inertial_coordinates =
        partial_derivative(inertial_coordinates, mesh, inv_jacobian);
    // Collect volume data
    // Remove tensor types, only storing individual components
    std::vector<TensorComponent> components;
    components.reserve(Dim + 1);
    for (size_t d = 0; d < Dim; d++) {
      components.emplace_back("InertialCoordinates_" +
                                  inertial_coordinates.component_name(
                                      inertial_coordinates.get_tensor_index(d)),
                              inertial_coordinates.get(d));
    }

    for (size_t i = 0; i < deriv_inertial_coordinates.size(); ++i) {
      components.emplace_back(
          "DerivInertialCoordinates_" +
              deriv_inertial_coordinates.component_name(
                  deriv_inertial_coordinates.get_tensor_index(i)),
          deriv_inertial_coordinates[i]);
    }

    // Also output the determinant of the inverse jacobian, which measures
    // the expansion and compression of the grid
    const auto& det_inv_jac = db::get<
        domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
        box);
    components.emplace_back(
        db::tag_name<domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                                  Frame::Inertial>>(),
        get(det_inv_jac));

    // Also output the jacobian diagnostic, which compares the analytic
    // Jacobian (via the CoordinateMap) to the numerical Jacobian
    // (computed via logical_partial_derivative)
    const auto& jacobian = determinant_and_inverse(inv_jacobian).second;
    tnsr::i<DataVector, Dim, Frame::ElementLogical> jac_diag{
        mesh.number_of_grid_points(), 0.0};
    domain::jacobian_diagnostic(make_not_null(&jac_diag), jacobian,
                                inertial_coordinates, mesh);
    for (size_t i = 0; i < Dim; ++i) {
      components.emplace_back(
          "JacobianDiagnostic_" +
              jac_diag.component_name(jac_diag.get_tensor_index(i)),
          jac_diag.get(i));
    }

    // Also output the computation domain metric
    const auto& flat_logical_metric =
        db::get<domain::Tags::FlatLogicalMetric<Dim>>(box);
    for (size_t i = 0; i < flat_logical_metric.size(); ++i) {
      components.emplace_back(
          db::tag_name<domain::Tags::FlatLogicalMetric<Dim>>() +
              flat_logical_metric.component_suffix(i),
          flat_logical_metric[i]);
    }

    // Send data to volume observer
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer, observers::ObservationId(time, "ObserveCoords"),
        std::string{"/element_data"},
        Parallel::make_array_component_id<ParallelComponent>(element_id),
        ElementVolumeData{element_id, std::move(components), mesh});
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct FindGlobalMinimumGridSpacing {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey("min_grid_spacing")};
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const double time = get<Tags::Time>(box);
    const double local_min_grid_spacing =
        get<domain::Tags::MinimumGridSpacing<Dim, Frame::Inertial>>(box);
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer, observers::ObservationId(time, "min_grid_spacing"),
        Parallel::make_array_component_id<ParallelComponent>(element_id),
        std::string{"/MinGridSpacing"},
        std::vector<std::string>{"Time", "MinGridSpacing"},
        MinGridSpacingReductionData{time, local_min_grid_spacing},
        std::make_optional(MinGridSpacingFormatter{}));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions

namespace Initialization {
template <size_t Dim>
struct SetMeshType {
  using const_global_cache_tags =
      tmpl::list<evolution::dg::subcell::Tags::ActiveGrid>;
  using mutable_global_cache_tags = tmpl::list<>;
  using argument_tags = tmpl::list<evolution::dg::subcell::Tags::ActiveGrid>;
  using simple_tags_from_options = tmpl::list<>;
  using default_initialized_simple_tags = tmpl::list<>;
  using return_tags = tmpl::list<domain::Tags::Mesh<Dim>>;
  using simple_tags = return_tags;
  using compute_tags = tmpl::list<>;

  static void apply(const gsl::not_null<::Mesh<Dim>*> mesh,
                    const evolution::dg::subcell::ActiveGrid active_grid) {
    // Originally the mesh is DG so switch it to FD if we aren't using a DG grid
    if (active_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
      *mesh = evolution::dg::subcell::fd::mesh(*mesh);
    }
  }
};
}  // namespace Initialization

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using TimeStepperBase = TimeStepper;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;

  // A placeholder system for the domain creators
  struct system {};

  static constexpr Options::String help{
      "Export the inertial coordinates of the Domain specified in the input "
      "file. The output can be used to compute initial data externally, for "
      "instance. Also outputs the determinant of the inverse jacobian as a "
      "diagnostic of Domain quality: values far from unity indicate "
      "compression or expansion of the grid."};

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            amr::Criterion,
            tmpl::list<amr::Criteria::DriveToTarget<volume_dim>,
                       amr::Criteria::Random,
                       amr::Criteria::TruncationError<
                           volume_dim, tmpl::list<::domain::Tags::Coordinates<
                                           volume_dim, Frame::Inertial>>>>>,
        tmpl::pair<Event, tmpl::list<Events::MonitorMemory<volume_dim>,
                                     Events::Completion>>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, tmpl::list<>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>, tmpl::list<>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::list<Triggers::Always, Triggers::SlabCompares,
                                       Triggers::TimeCompares>>>;
  };

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterWithObservers<
                     Actions::ExportCoordinates<Dim>>,
                 observers::Actions::RegisterWithObservers<
                     Actions::FindGlobalMinimumGridSpacing>>;

  using dg_element_array = DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<
              Parallel::Phase::Initialization,
              tmpl::list<Initialization::Actions::InitializeItems<
                             Initialization::TimeStepping<Metavariables,
                                                          TimeStepperBase>,
                             evolution::dg::Initialization::Domain<Dim>,
                             ::amr::Initialization::Initialize<volume_dim>,
                             Initialization::SetMeshType<Dim>>,
                         Initialization::Actions::AddComputeTags<tmpl::list<
                             ::domain::Tags::MinimumGridSpacingCompute<
                                 Dim, Frame::Inertial>,
                             ::domain::Tags::FlatLogicalMetricCompute<Dim>>>,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::push_back<dg_registration_list,
                              Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Execute,
              tmpl::list<
                  Actions::AdvanceTime, Actions::ExportCoordinates<Dim>,
                  Actions::FindGlobalMinimumGridSpacing,
                  evolution::Actions::RunEventsAndTriggers<local_time_stepping>,
                  PhaseControl::Actions::ExecutePhaseChange>>>>;

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = dg_element_array;
    using projectors = tmpl::list<
        Initialization::ProjectTimeStepping<volume_dim>,
        evolution::dg::Initialization::ProjectDomain<volume_dim>,
        ::amr::projectors::DefaultInitialize<
            Initialization::Tags::InitialTimeDelta,
            Initialization::Tags::InitialSlabSize<local_time_stepping>,
            ::domain::Tags::InitialExtents<Dim>,
            ::domain::Tags::InitialRefinementLevels<Dim>,
            evolution::dg::Tags::Quadrature>>;
  };

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, dg_registration_list>>;
  };

  using component_list =
      tmpl::list<::amr::Component<Metavariables>, dg_element_array,
                 mem_monitor::MemoryMonitor<Metavariables>,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>;

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<MinGridSpacingReductionData>>;

  static constexpr auto default_phase_order =
      std::array{Parallel::Phase::Initialization, Parallel::Phase::Register,
                 Parallel::Phase::CheckDomain, Parallel::Phase::Execute,
                 Parallel::Phase::Exit};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
/// \endcond
