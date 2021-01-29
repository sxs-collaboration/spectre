// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.tpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/GrTagsForHydro.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Initialize.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1HydroCoupling.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderSchemeLts.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  static constexpr size_t volume_dim = 3;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;

  // To switch which initial data is evolved you only need to change the
  // line `using initial_data = ...;` and include the header file for the
  // solution.
  using initial_data = RadiationTransport::M1Grey::Solutions::ConstantM1;
  static_assert(
      evolution::is_analytic_data_v<initial_data> xor
          evolution::is_analytic_solution_v<initial_data>,
      "initial_data must be either an analytic_data or an analytic_solution");

  // Set list of neutrino species to be used by M1 code
  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>>;

  using system = RadiationTransport::M1Grey::System<neutrino_species>;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;
  using initial_data_tag =
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;
  using boundary_condition_tag = initial_data_tag;
  using analytic_variables_tags = typename system::variables_tag::tags_list;
  using normal_dot_numerical_flux =
      Tags::NumericalFlux<dg::NumericalFluxes::LocalLaxFriedrichs<system>>;
  using limiter = Tags::Limiter<
      Limiters::Minmod<3, typename system::variables_tag::tags_list>>;

  using step_choosers_common =
      tmpl::list<//StepChoosers::Registrars::Cfl<volume_dim, Frame::Inertial>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;
  using step_choosers_for_step_only =
      tmpl::list<StepChoosers::Registrars::PreventRapidIncrease>;
  using step_choosers_for_slab_only =
      tmpl::list<StepChoosers::Registrars::StepToTimes>;
  using step_choosers = tmpl::conditional_t<
      local_time_stepping,
      tmpl::append<step_choosers_common, step_choosers_for_step_only>,
      tmpl::list<>>;
  using slab_choosers = tmpl::conditional_t<
      local_time_stepping,
      tmpl::append<step_choosers_common, step_choosers_for_slab_only>,
      tmpl::append<step_choosers_common, step_choosers_for_step_only,
                   step_choosers_for_slab_only>>;

  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;
  using boundary_scheme = tmpl::conditional_t<
      local_time_stepping,
      dg::FirstOrderScheme::FirstOrderSchemeLts<
          volume_dim, typename system::variables_tag,
          db::add_tag_prefix<::Tags::dt, typename system::variables_tag>,
          normal_dot_numerical_flux, Tags::TimeStepId, time_stepper_tag>,
      dg::FirstOrderScheme::FirstOrderScheme<
          volume_dim, typename system::variables_tag,
          db::add_tag_prefix<::Tags::dt, typename system::variables_tag>,
          normal_dot_numerical_flux, Tags::TimeStepId>>;

  // public for use by the Charm++ registration code
  using events = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          dg::Events::Registrars::ObserveErrorNorms<
                              Tags::Time, analytic_variables_tags>,
                          tmpl::list<>>,
      dg::Events::Registrars::ObserveFields<
          3, Tags::Time,
          tmpl::append<typename system::variables_tag::tags_list,
                       typename system::primitive_variables_tag::tags_list>,
          tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                              analytic_variables_tags, tmpl::list<>>>,
      Events::Registrars::ObserveTimeStep<EvolutionMetavars>,
      Events::Registrars::ChangeSlabSize<slab_choosers>>>;
  using triggers = Triggers::time_triggers;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      typename Event<events>::creatable_classes>;

  using step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data>,
          dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
          tmpl::list<>>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme,
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      tmpl::conditional_t<local_time_stepping,
                          tmpl::list<Actions::RecordTimeStepperData<>,
                                     Actions::MutateApply<boundary_scheme>>,
                          tmpl::list<Actions::MutateApply<boundary_scheme>,
                                     Actions::RecordTimeStepperData<>>>,
      Actions::UpdateU<>, Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>,
      Actions::MutateApply<typename RadiationTransport::M1Grey::
                               ComputeM1Closure<neutrino_species>>,
      Actions::MutateApply<typename RadiationTransport::M1Grey::
                               ComputeM1HydroCoupling<neutrino_species>>>>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    RegisterWithObserver,
    Evolve,
    Exit
  };

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<volume_dim>,
      Initialization::Actions::GrTagsForHydro<system>,
      Initialization::Actions::ConservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<volume_dim, Frame::Logical>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      RadiationTransport::M1Grey::Actions::InitializeM1Tags<system>,
      Actions::MutateApply<typename RadiationTransport::M1Grey::
                               ComputeM1Closure<neutrino_species>>,
      Actions::MutateApply<typename RadiationTransport::M1Grey::
                               ComputeM1HydroCoupling<neutrino_species>>,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag,
              typename system::spacetime_variables_tag,
              typename system::primitive_variables_tag>,
          dg::Initialization::slice_tags_to_exterior<
              typename system::spacetime_variables_tag,
              typename system::primitive_variables_tag>,
          dg::Initialization::face_compute_tags<>,
          dg::Initialization::exterior_compute_tags<>, true, true>,
      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data>,
          Initialization::Actions::AddComputeTags<
              tmpl::list<evolution::Tags::AnalyticCompute<
                  3, initial_data_tag, analytic_variables_tags>>>,
          tmpl::list<>>,
      dg::Actions::InitializeMortars<boundary_scheme>,
      Initialization::Actions::DiscontinuousGalerkin<EvolutionMetavars>,
      Initialization::Actions::Minmod<3>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      DgElementArray<
          EvolutionMetavars,
          tmpl::list<
              Parallel::PhaseActions<Phase, Phase::Initialization,
                                     initialization_actions>,

              Parallel::PhaseActions<
                  Phase, Phase::InitializeTimeStepperHistory,
                  SelfStart::self_start_procedure<step_actions>>,

              Parallel::PhaseActions<
                  Phase, Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterEventsWithObservers,
                             Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::list<
                      Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                      tmpl::conditional_t<
                          local_time_stepping,
                          Actions::ChangeStepSize<step_choosers>, tmpl::list<>>,
                      step_actions, Actions::AdvanceTime>>>>>;

  using const_global_cache_tags =
      tmpl::list<initial_data_tag, normal_dot_numerical_flux, time_stepper_tag,
                 Tags::EventsAndTriggers<events, triggers>>;

  static constexpr Options::String help{
      "Evolve the M1Grey system (without coupling to hydro).\n\n"};

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
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
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::slab_choosers>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeSequence<double>>,
    &Parallel::register_derived_classes_with_charm<TimeSequence<std::uint64_t>>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
