// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"            // IWYU pragma: keep
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/FilterAction.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveGr.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/ChangeSlabSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"       // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/StepChoosers/ByBlock.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

template <size_t Dim, typename InitialData>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;
  // Customization/"input options" to simulation
  using initial_data_tag =
      tmpl::conditional_t<evolution::is_analytic_solution_v<InitialData>,
                          Tags::AnalyticSolution<InitialData>,
                          Tags::AnalyticData<InitialData>>;
  static_assert(
      evolution::is_analytic_data_v<InitialData> xor
          evolution::is_analytic_solution_v<InitialData>,
      "initial_data must be either an analytic_data or an analytic_solution");

  using system = CurvedScalarWave::System<Dim>;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = true;
  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;

  using observe_fields = tmpl::append<
      typename system::variables_tag::tags_list,
      tmpl::list<::Tags::PointwiseL2Norm<
                     CurvedScalarWave::Tags::OneIndexConstraint<volume_dim>>,
                 ::Tags::PointwiseL2Norm<
                     CurvedScalarWave::Tags::TwoIndexConstraint<volume_dim>>>>;
  using analytic_solution_fields = typename system::variables_tag::tags_list;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<Events::Completion,
                                            dg::Events::field_observations<
                                                volume_dim, Tags::Time,
                                                observe_fields, tmpl::list<>>,
                                            Events::time_events<system>>>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>,
        tmpl::pair<
            CurvedScalarWave::BoundaryConditions::BoundaryCondition<volume_dim>,
            CurvedScalarWave::BoundaryConditions::standard_boundary_conditions<
                volume_dim>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::push_back<StepChoosers::standard_step_choosers<system>,
                                   StepChoosers::ByBlock<
                                       StepChooserUse::LtsStep, volume_dim>>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::push_back<StepChoosers::standard_slab_choosers<
                                       system, local_time_stepping>,
                                   StepChoosers::ByBlock<StepChooserUse::Slab,
                                                         volume_dim>>>,
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

  // The scalar wave system generally does not require filtering, except
  // possibly on certain deformed domains.  Here a filter is added in 2D for
  // testing purposes.  When performing numerical experiments with the scalar
  // wave system, the user should determine whether this filter can be removed.
  static constexpr bool use_filtering = (2 == volume_dim);

  using step_actions = tmpl::flatten<tmpl::list<
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
      tmpl::conditional_t<
          use_filtering,
          dg::Actions::Filter<
              Filters::Exponential<0>,
              tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Psi,
                         CurvedScalarWave::Phi<Dim>>>,
          tmpl::list<>>>>;

  enum class Phase {
    Initialization,
    RegisterWithObserver,
    InitializeTimeStepperHistory,
    LoadBalancing,
    WriteCheckpoint,
    Evolve,
    Exit
  };

  static std::string phase_name(Phase phase) noexcept {
    if (phase == Phase::LoadBalancing) {
      return "LoadBalancing";
    }
    ERROR(
        "Passed phase that should not be used in input file. Integer "
        "corresponding to phase is: "
        << static_cast<int>(phase));
  }

  using phase_changes =
      tmpl::list<PhaseControl::Registrars::VisitAndReturn<EvolutionMetavars,
                                                          Phase::LoadBalancing>,
                 PhaseControl::Registrars::CheckpointAndExitAfterWallclock<
                     EvolutionMetavars>>;

  using initialize_phase_change_decision_data =
      PhaseControl::InitializePhaseChangeDecisionData<phase_changes>;

  using phase_change_tags_and_combines_list =
      PhaseControl::get_phase_change_tags<phase_changes>;

  using const_global_cache_tags =
      tmpl::list<initial_data_tag, Tags::EventsAndTriggers,
                 PhaseControl::Tags::PhaseChangeAndTriggers<phase_changes>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<volume_dim>,
      Initialization::Actions::NonconservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<Dim, Frame::Logical>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      CurvedScalarWave::Actions::InitializeConstraintDampingGammas<volume_dim>,
      CurvedScalarWave::Actions::InitializeGrVars<volume_dim>,
      Initialization::Actions::AddComputeTags<tmpl::push_back<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>,
          evolution::Tags::AnalyticCompute<Dim, initial_data_tag,
                                           analytic_solution_fields>>>,
      CurvedScalarWave::Actions::InitializeConstraints<volume_dim>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Phase, Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<
                  Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                  step_actions, Actions::AdvanceTime,
                  PhaseControl::Actions::ExecutePhaseChange<phase_changes>>>>>;

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
      "Evolve a Scalar Wave in Dim spatial dimension.\n\n"
      "The numerical flux is:    UpwindFlux\n"};

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<EvolutionMetavars>&
          cache_proxy) noexcept {
    const auto next_phase = PhaseControl::arbitrate_phase_change<phase_changes>(
        phase_change_decision_data, current_phase,
        *(cache_proxy.ckLocalBranch()));
    if (next_phase.has_value()) {
      return next_phase.value();
    }
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
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
    &CurvedScalarWave::BoundaryCorrections::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
