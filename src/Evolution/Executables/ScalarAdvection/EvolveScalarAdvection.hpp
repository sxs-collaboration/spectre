// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/NeighborTciDecision.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/ScalarAdvection/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/GhostData.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOptions.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/VelocityAtFace.hpp"
#include "Evolution/Systems/ScalarAdvection/System.hpp"
#include "Evolution/Systems/ScalarAdvection/VelocityField.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Krivodonova.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Kuzmin.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Sinusoid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/ChangeSlabSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"       // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

template <size_t Dim>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;
  using system = ScalarAdvection::System<Dim>;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  // The use_dg_subcell flag controls whether to use "standard" limiting (false)
  // or a DG-FD hybrid scheme (true).
  static constexpr bool use_dg_subcell = true;

  using initial_data_list =
      tmpl::conditional_t<Dim == 1,
                          tmpl::list<ScalarAdvection::Solutions::Krivodonova,
                                     ScalarAdvection::Solutions::Sinusoid>,
                          tmpl::list<ScalarAdvection::Solutions::Kuzmin>>;

  using limiter = Tags::Limiter<
      Limiters::Minmod<Dim, typename system::variables_tag::tags_list>>;

  using analytic_variables_tags = typename system::variables_tag::tags_list;
  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_variables_tags, use_dg_subcell, initial_data_list>;
  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;
  using observe_fields = tmpl::push_back<
      tmpl::append<
          typename system::variables_tag::tags_list, error_tags,
          tmpl::conditional_t<use_dg_subcell,
                              tmpl::list<evolution::dg::subcell::Tags::
                                             TciStatusCompute<volume_dim>>,
                              tmpl::list<>>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<volume_dim,
                                                                   Frame::Grid>,
          domain::Tags::Coordinates<volume_dim, Frame::Grid>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::Inertial>,
          domain::Tags::Coordinates<volume_dim, Frame::Inertial>>>;
  using non_tensor_compute_tags = tmpl::list<
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverMeshCompute<volume_dim>,
          ::Events::Tags::ObserverMeshCompute<volume_dim>>,
      analytic_compute, error_compute>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<Event, tmpl::flatten<tmpl::list<
                              Events::Completion,
                              dg::Events::field_observations<
                                  volume_dim, Tags::Time, observe_fields,
                                  non_tensor_compute_tags>,
                              Events::time_events<system>>>>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<ScalarAdvection::BoundaryConditions::BoundaryCondition<Dim>,
                   ScalarAdvection::BoundaryConditions::
                       standard_boundary_conditions<Dim>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            StepChoosers::standard_slab_choosers<system, local_time_stepping>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  struct SubcellOptions {
    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = false;

    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<ScalarAdvection::fd::Tags::Reconstructor<volume_dim>>(box)
          .ghost_zone_size();
    }

    using DgComputeSubcellNeighborPackagedData =
        ScalarAdvection::subcell::NeighborPackagedData;

    using GhostVariables = ScalarAdvection::subcell::GhostVariables;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  using dg_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          system, volume_dim, false>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<
              Actions::RecordTimeStepperData<system>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<system>>>,
      Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>>>;

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,
      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,

      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          system, volume_dim, false>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<
              Actions::RecordTimeStepperData<system>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<system>>>,
      evolution::dg::subcell::Actions::TciAndRollback<
          ScalarAdvection::subcell::TciOnDgGrid<Dim>>,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,
      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim, ScalarAdvection::subcell::GhostVariables,
          local_time_stepping>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          ScalarAdvection::subcell::TimeDerivative<volume_dim>>,
      Actions::RecordTimeStepperData<system>, Actions::UpdateU<system>,
      evolution::dg::subcell::Actions::TciAndSwitchToDg<
          ScalarAdvection::subcell::TciOnFdGrid<volume_dim>>,
      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData,
                 tmpl::conditional_t<
                     use_dg_subcell,
                     tmpl::list<ScalarAdvection::fd::Tags::Reconstructor<Dim>,
                                ScalarAdvection::subcell::Tags::TciOptions>,
                     tmpl::list<>>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, local_time_stepping>,
          evolution::dg::Initialization::Domain<volume_dim>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::ConservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,

      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::Initialize<
                  volume_dim, system,
                  ScalarAdvection::subcell::DgInitialDataTci<volume_dim>>,
              Initialization::Actions::AddSimpleTags<
                  ScalarAdvection::subcell::VelocityAtFace<volume_dim>>,
              Actions::MutateApply<
                  ScalarAdvection::subcell::SetInitialRdmpData>>,
          tmpl::list<>>,

      Initialization::Actions::AddComputeTags<
          tmpl::list<ScalarAdvection::Tags::VelocityFieldCompute<Dim>>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars,
                                                  local_time_stepping>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      Initialization::Actions::Minmod<Dim>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type =
        std::conditional_t<std::is_same_v<ParallelComponent, dg_element_array>,
                           dg_registration_list, tmpl::list<>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array>;

  static constexpr Options::String help{
      "Evolve the scalar advection equation.\n\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &ScalarAdvection::BoundaryCorrections::register_derived_with_charm,
    &ScalarAdvection::fd::register_derived_with_charm,
    &register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
