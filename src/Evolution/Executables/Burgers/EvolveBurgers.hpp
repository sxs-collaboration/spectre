// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Domain/Actions/InitializeDomain.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"    // IWYU pragma: keep
#include "Evolution/Actions/InitializeConservativeSystem.hpp"
#include "Evolution/Actions/InitializeEvolution.hpp"
#include "Evolution/Actions/InitializeLimiter.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/ObserveErrorNorms.hpp"
#include "Evolution/DiscontinuousGalerkin/ObserveFields.hpp"
#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Tags.hpp"
#include "Evolution/Systems/Burgers/Equations.hpp"  // IWYU pragma: keep // for LocalLaxFriedrichsFlux
#include "Evolution/Systems/Burgers/System.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeInterfaces.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"       // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/StepChoosers/Cfl.hpp"               // IWYU pragma: keep
#include "Time/StepChoosers/Constant.hpp"          // IWYU pragma: keep
#include "Time/StepChoosers/Increase.hpp"          // IWYU pragma: keep
#include "Time/StepChoosers/StepChooser.hpp"
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
  using system = Burgers::System;
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = false;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<Burgers::Solutions::Step>;
  using boundary_condition_tag = analytic_solution_tag;
  using normal_dot_numerical_flux =
      OptionTags::NumericalFlux<Burgers::LocalLaxFriedrichsFlux>;
  using limiter = OptionTags::Limiter<
      Limiters::Minmod<1, system::variables_tag::tags_list>>;

  // public for use by the Charm++ registration code
  using events =
      tmpl::list<dg::Events::Registrars::ObserveFields<
                     1, db::get_variables_tags_list<system::variables_tag>,
                     db::get_variables_tags_list<system::variables_tag>>,
                 dg::Events::Registrars::ObserveErrorNorms<
                     1, db::get_variables_tags_list<system::variables_tag>>>;
  using triggers = Triggers::time_triggers;

  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>,
                 OptionTags::EventsAndTriggers<events, triggers>>;
  using domain_creator_tag = OptionTags::DomainCreator<1, Frame::Inertial>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<Event<events>::creatable_classes>;

  using step_choosers =
      tmpl::list<StepChoosers::Registrars::Cfl<1, Frame::Inertial>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      Actions::ComputeVolumeFluxes,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeTimeDerivative,
      dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      Actions::RecordTimeStepperData>>;
  using update_variables = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      Actions::UpdateU, Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>>>;

  enum class Phase {
    Initialization,
    RegisterWithObserver,
    InitializeTimeStepperHistory,
    Evolve,
    Exit
  };

  using initialization_actions =
      tmpl::list<domain::Actions::InitializeDomain<1>,
                 evolution::Actions::InitializeConservativeSystem,
                 dg::Actions::InitializeInterfaces<
                     system,
                     dg::Initialization::slice_tags_to_face<
                         typename system::variables_tag>,
                     dg::Initialization::slice_tags_to_exterior<>>,
                 evolution::Actions::InitializeEvolution<system>,
                 dg::Actions::InitializeMortars<EvolutionMetavars>,
                 dg::Actions::InitializeFluxes<EvolutionMetavars>,
                 evolution::Actions::InitializeMinMod<1>,
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
                                     element_observation_type>>,
                             Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::InitializeTimeStepperHistory,
                  tmpl::flatten<tmpl::list<SelfStart::self_start_procedure<
                      compute_rhs, update_variables>>>>,

              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::flatten<tmpl::list<
                      Actions::AdvanceTime, Actions::RunEventsAndTriggers,
                      tmpl::conditional_t<
                          local_time_stepping,
                          Actions::ChangeStepSize<step_choosers>, tmpl::list<>>,
                      compute_rhs, update_variables>>>>,
          Parallel::ForwardAllOptionsToDataBox<
              Initialization::option_tags<initialization_actions>>>>;

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
    &setup_error_handling, &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
