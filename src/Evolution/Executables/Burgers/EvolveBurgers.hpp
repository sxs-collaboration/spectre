// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory1D.hpp"
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
#include "Evolution/DgSubcell/Events/ObserveFields.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
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
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Subcell/GhostData.hpp"
#include "Evolution/Systems/Burgers/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/Burgers/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/Burgers/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/Burgers/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/Burgers/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticData/Burgers/Sinusoid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Bump.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"                // IWYU pragma: keep
#include "Time/Actions/ChangeSlabSize.hpp"             // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"             // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"      // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"           // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                    // IWYU pragma: keep
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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

template <typename InitialData>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = 1;
  using system = Burgers::System;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  // The use_dg_subcell flag controls whether to use "standard" limiting (false)
  // or a DG-FD hybrid scheme (true).
  static constexpr bool use_dg_subcell = true;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;

  using initial_data = InitialData;
  static_assert(
      evolution::is_analytic_data_v<initial_data> xor
          evolution::is_analytic_solution_v<initial_data>,
      "initial_data must be either an analytic_data or an analytic_solution");
  using initial_data_tag =
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;

  using limiter =
      Tags::Limiter<Limiters::Minmod<1, system::variables_tag::tags_list>>;

  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;

  using observe_fields = typename system::variables_tag::tags_list;
  using analytic_solution_fields =
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          observe_fields, tmpl::list<>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                tmpl::conditional_t<
                    use_dg_subcell,
                    evolution::dg::subcell::Events::ObserveFields<
                        volume_dim, Tags::Time,
                        tmpl::push_back<observe_fields, evolution::dg::subcell::
                                                            Tags::TciStatus>,
                        analytic_solution_fields>,
                    dg::Events::field_observations<volume_dim, Tags::Time,
                                                   observe_fields,
                                                   analytic_solution_fields>>,
                Events::time_events<system>>>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            StepChoosers::standard_slab_choosers<system, local_time_stepping>>,
        tmpl::pair<StepController, StepControllers::standard_step_controllers>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    Register,
    LoadBalancing,
    WriteCheckpoint,
    Evolve,
    Exit
  };

  static std::string phase_name(Phase phase) noexcept {
    if (phase == Phase::LoadBalancing) {
      return "LoadBalancing";
    } else if (phase == Phase::WriteCheckpoint) {
      return "WriteCheckpoint";
    }
    ERROR(
        "Passed phase that should not be used in input file. Integer "
        "corresponding to phase is: "
        << static_cast<int>(phase));
  }

  using phase_changes =
      tmpl::list<PhaseControl::Registrars::VisitAndReturn<EvolutionMetavars,
                                                          Phase::LoadBalancing>,
                 PhaseControl::Registrars::VisitAndReturn<
                     EvolutionMetavars, Phase::WriteCheckpoint>,
                 PhaseControl::Registrars::CheckpointAndExitAfterWallclock<
                     EvolutionMetavars>>;

  using initialize_phase_change_decision_data =
      PhaseControl::InitializePhaseChangeDecisionData<phase_changes>;

  using phase_change_tags_and_combines_list =
      PhaseControl::get_phase_change_tags<phase_changes>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::flatten<tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<volume_dim>,
      Initialization::Actions::ConservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<volume_dim, Frame::Logical>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,

      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<evolution::dg::subcell::Actions::Initialize<
              volume_dim, system, Burgers::subcell::DgInitialDataTci>>,
          tmpl::list<>>,

      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data>,
          Initialization::Actions::AddComputeTags<
              tmpl::list<evolution::Tags::AnalyticCompute<
                  volume_dim, initial_data_tag, analytic_solution_fields>>>,
          tmpl::list<>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      Initialization::Actions::Minmod<volume_dim>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>>;

  using dg_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<>,
                     evolution::dg::Actions::ApplyBoundaryCorrections<
                         EvolutionMetavars>>,
          tmpl::list<evolution::dg::Actions::ApplyBoundaryCorrections<
                         EvolutionMetavars>,
                     Actions::RecordTimeStepperData<>,
                     evolution::Actions::RunEventsAndDenseTriggers<>,
                     Actions::UpdateU<>>>,
      Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>>>;

  struct SubcellOptions {
    static constexpr bool subcell_enabled = use_dg_subcell;
    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) noexcept {
      return db::get<Burgers::fd::Tags::Reconstructor>(box).ghost_zone_size();
    }

    using DgComputeSubcellNeighborPackagedData =
        Burgers::subcell::NeighborPackagedData;

    using GhostDataToSlice = Burgers::subcell::GhostDataToSlice;
  };

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      evolution::dg::Actions::ApplyBoundaryCorrections<EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<Actions::RecordTimeStepperData<>, Actions::UpdateU<>>>,
      evolution::dg::subcell::Actions::TciAndRollback<
          Burgers::subcell::TciOnDgGrid>,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim, Burgers::subcell::GhostDataOnSubcells>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          Burgers::subcell::TimeDerivative>,
      Actions::RecordTimeStepperData<>, Actions::UpdateU<>,
      evolution::dg::subcell::Actions::TciAndSwitchToDg<
          Burgers::subcell::TciOnFdGrid>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using dg_element_array_component = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Phase, Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<
              Phase, Phase::Register,
              tmpl::push_back<dg_registration_list,
                              Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<
                  Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                  step_actions, Actions::AdvanceTime,
                  PhaseControl::Actions::ExecutePhaseChange<phase_changes>>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, dg_element_array_component>,
        dg_registration_list, tmpl::list<>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array_component>;

  using const_global_cache_tags = tmpl::push_back<
      tmpl::conditional_t<use_dg_subcell,
                          tmpl::list<Burgers::fd::Tags::Reconstructor>,
                          tmpl::list<>>,
      initial_data_tag, Tags::EventsAndTriggers,
      PhaseControl::Tags::PhaseChangeAndTriggers<phase_changes>>;

  static constexpr Options::String help{"Evolve the Burgers equation.\n\n"};

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<EvolutionMetavars>&
          cache_proxy) noexcept {
    const auto next_phase =
        PhaseControl::arbitrate_phase_change<phase_changes>(
            phase_change_decision_data, current_phase,
            *(cache_proxy.ckLocalBranch()));
    if (next_phase.has_value()) {
      return next_phase.value();
    }
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::Register;
      case Phase::Register:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &Burgers::BoundaryConditions::register_derived_with_charm,
    &Burgers::BoundaryCorrections::register_derived_with_charm,
    &Burgers::fd::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

std::ostream& operator<<(
    std::ostream& os,
    const typename EvolutionMetavars<Burgers::Solutions::Step>::Phase&
        phase) noexcept {
  using metavars = EvolutionMetavars<Burgers::Solutions::Step>;
  switch (phase) {
    case metavars::Phase::Initialization:
      os << "Initialization";
      break;
    case metavars::Phase::InitializeTimeStepperHistory:
      os << "InitializeTimeStepperHistory";
      break;
    case metavars::Phase::Register:
      os << "Register";
      break;
    case metavars::Phase::LoadBalancing:
      os << "LoadBalancing";
      break;
    case metavars::Phase::WriteCheckpoint:
      os << "WriteCheckpoint";
      break;
    case metavars::Phase::Evolve:
      os << "Evolve";
      break;
    case metavars::Phase::Exit:
      os << "Exit";
      break;
    default:
      ERROR("Unknown phase passed to stream operator.");
  }
  return os;
}
