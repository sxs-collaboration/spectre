// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/AddMeshVelocitySourceTerms.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"    // IWYU pragma: keep
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Conservative/ConservativeDuDt.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"                // IWYU pragma: keep
#include "Time/Actions/ChangeSlabSize.hpp"             // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"             // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"      // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"           // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                    // IWYU pragma: keep
#include "Time/StepChoosers/Cfl.hpp"                   // IWYU pragma: keep
#include "Time/StepChoosers/Constant.hpp"              // IWYU pragma: keep
#include "Time/StepChoosers/Increase.hpp"              // IWYU pragma: keep
#include "Time/StepChoosers/PreventRapidIncrease.hpp"  // IWYU pragma: keep
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  static constexpr size_t volume_dim = 1;
  using system = Burgers::System;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;
  using initial_data_tag = Tags::AnalyticSolution<Burgers::Solutions::Step>;
  using boundary_condition_tag = initial_data_tag;
  using normal_dot_numerical_flux =
      Tags::NumericalFlux<dg::NumericalFluxes::LocalLaxFriedrichs<system>>;
  using limiter =
      Tags::Limiter<Limiters::Minmod<1, system::variables_tag::tags_list>>;

  using step_choosers_common =
      tmpl::list<StepChoosers::Registrars::Cfl<volume_dim, Frame::Inertial>,
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

  // public for use by the Charm++ registration code
  using observe_fields =
      db::get_variables_tags_list<typename system::variables_tag>;
  using analytic_solution_fields = observe_fields;
  using events =
      tmpl::list<dg::Events::Registrars::ObserveFields<
                     1, Tags::Time, observe_fields, analytic_solution_fields>,
                 dg::Events::Registrars::ObserveErrorNorms<
                     Tags::Time, analytic_solution_fields>,
                 Events::Registrars::ChangeSlabSize<slab_choosers>>;
  using triggers = Triggers::time_triggers;

  using const_global_cache_tags =
      tmpl::list<initial_data_tag,
                 Tags::TimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>,
                 Tags::EventsAndTriggers<events, triggers>>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<Event<events>::creatable_classes>;

  using step_actions = tmpl::flatten<tmpl::list<
      Actions::ComputeVolumeFluxes,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeTimeDerivative<
          evolution::dg::ConservativeDuDt<Burgers::System, dg_formulation>>,
      evolution::Actions::AddMeshVelocitySourceTerms,
      dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      Actions::RecordTimeStepperData<>,
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      Actions::UpdateU<>, Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>>>;

  enum class Phase {
    Initialization,
    RegisterWithObserver,
    InitializeTimeStepperHistory,
    Evolve,
    Exit
  };

  using initialization_actions = tmpl::list<
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<1>,
      Initialization::Actions::ConservativeSystem,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag>,
          dg::Initialization::slice_tags_to_exterior<>,
          dg::Initialization::face_compute_tags<>,
          dg::Initialization::exterior_compute_tags<>, true, true>,
      Initialization::Actions::AddComputeTags<
          tmpl::list<evolution::Tags::AnalyticCompute<
              1, initial_data_tag, analytic_solution_fields>>>,
      dg::Actions::InitializeMortars<EvolutionMetavars>,
      Initialization::Actions::DiscontinuousGalerkin<EvolutionMetavars>,
      Initialization::Actions::Minmod<1>,
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
                  Phase, Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterWithObservers<
                                 observers::RegisterObservers<
                                     Tags::Time, element_observation_type>>,
                             Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::InitializeTimeStepperHistory,
                  SelfStart::self_start_procedure<step_actions>>,

              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::list<
                      Actions::RunEventsAndTriggers,
                      Actions::ChangeSlabSize,
                      tmpl::conditional_t<
                          local_time_stepping,
                          Actions::ChangeStepSize<step_choosers>, tmpl::list<>>,
                      step_actions, Actions::AdvanceTime>>>>>;

  static constexpr OptionString help{
      "Evolve the Burgers equation.\n\n"
      "The analytic solution is: Linear\n"
      "The numerical flux is:    LocalLaxFriedrichs\n"};

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
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
    &setup_error_handling,
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
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
