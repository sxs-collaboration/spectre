// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Filtering.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/InitializeElement.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/ObserveErrorNorms.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/ObserveFields.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/Equations.hpp"  // IWYU pragma: keep // for UpwindFlux
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "IO/Observer/Actions.hpp"            // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"            // IWYU pragma: keep
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/GotoAction.hpp"  // IWYU pragma: keep
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/FinalTime.hpp"              // IWYU pragma: keep
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
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

template <size_t Dim>
struct EvolutionMetavars {
  // Customization/"input options" to simulation
  using system = ScalarWave::System<Dim>;
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = true;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<ScalarWave::Solutions::PlaneWave<Dim>>;
  using boundary_condition_tag = analytic_solution_tag;
  using normal_dot_numerical_flux =
      OptionTags::NumericalFluxParams<ScalarWave::UpwindFlux<Dim>>;

  // public for use by the Charm++ registration code
  using events = tmpl::list<
      dg::Events::Registrars::ObserveFields<
          Dim, db::get_variables_tags_list<typename system::variables_tag>,
          db::get_variables_tags_list<typename system::variables_tag>>,
      dg::Events::Registrars::ObserveErrorNorms<
          Dim, db::get_variables_tags_list<typename system::variables_tag>>>;
  using triggers = Triggers::time_triggers;

  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>,
                 OptionTags::EventsAndTriggers<events, triggers>>;
  using domain_creator_tag = OptionTags::DomainCreator<Dim, Frame::Inertial>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      typename Event<events>::creatable_classes>;

  using step_choosers =
      tmpl::list<StepChoosers::Registrars::Cfl<Dim, Frame::Inertial>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::InternalDirections<Dim>>,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeTimeDerivative,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::BoundaryDirectionsInterior<Dim>>,
      dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      Actions::RecordTimeStepperData>>;
  // To add filtering to the executable add the action:
  //
  // dg::Actions::ExponentialFilter<0,
  //        tmpl::list<ScalarWave::Pi, ScalarWave::Psi, ScalarWave::Phi<Dim>>>
  //
  // to the end of `update_variables` so it is after `Actions::UpdateU`.
  using update_variables = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      Actions::UpdateU>>;

  struct EvolvePhaseStart;
  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      DgElementArray<
          EvolutionMetavars, dg::Actions::InitializeElement<Dim>,
          tmpl::flatten<tmpl::list<
              SelfStart::self_start_procedure<compute_rhs, update_variables>,
              Actions::Label<EvolvePhaseStart>, Actions::AdvanceTime,
              Actions::RunEventsAndTriggers, Actions::FinalTime,
              tmpl::conditional_t<local_time_stepping,
                                  Actions::ChangeStepSize<step_choosers>,
                                  tmpl::list<>>,
              compute_rhs, update_variables,
              Actions::Goto<EvolvePhaseStart>>>>>;

  static constexpr OptionString help{
      "Evolve a Scalar Wave in Dim spatial dimension.\n\n"
      "The analytic solution is: PlaneWave\n"
      "The numerical flux is:    UpwindFlux\n"};

  enum class Phase {
    Initialization,
    RegisterWithObserver,
    Evolve,
    Exit
  };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
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
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
