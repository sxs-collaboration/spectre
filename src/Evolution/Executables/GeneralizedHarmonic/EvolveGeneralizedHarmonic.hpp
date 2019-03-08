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
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"  // IWYU pragma: keep // for UpwindFlux
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Observe.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"               // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/GotoAction.hpp"  // IWYU pragma: keep
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrapGh.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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

struct EvolutionMetavars {
  // Customization/"input options" to simulation
  static constexpr int dim = 3;
  using Inertial = Frame::Inertial;
  using system = GeneralizedHarmonic::System<dim>;
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = false;
  using analytic_solution_tag = OptionTags::AnalyticSolution<
      GeneralizedHarmonic::Solutions::WrapGh<gr::Solutions::KerrSchild>>;
  using boundary_condition_tag = analytic_solution_tag;
  using normal_dot_numerical_flux = OptionTags::NumericalFluxParams<
      // dg::NumericalFluxes::LocalLaxFriedrichs<system>>;
      GeneralizedHarmonic::UpwindFlux<dim>>;
  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>>;
  using domain_creator_tag = OptionTags::DomainCreator<dim, Inertial>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<GeneralizedHarmonic::Actions::Observe>>;

  using step_choosers = tmpl::list<StepChoosers::Registrars::Cfl<dim, Inertial>,
                                   StepChoosers::Registrars::Constant,
                                   StepChoosers::Registrars::Increase>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::InternalDirections<dim>>,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeTimeDerivative,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::BoundaryDirectionsInterior<dim>>,
      dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      Actions::RecordTimeStepperData>>;
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
          EvolutionMetavars, GeneralizedHarmonic::Actions::Initialize<dim>,
          tmpl::flatten<tmpl::list<
              SelfStart::self_start_procedure<compute_rhs, update_variables>,
              Actions::Label<EvolvePhaseStart>, Actions::AdvanceTime,
              GeneralizedHarmonic::Actions::Observe, Actions::FinalTime,
              tmpl::conditional_t<local_time_stepping,
                                  Actions::ChangeStepSize<step_choosers>,
                                  tmpl::list<>>,
              compute_rhs, update_variables,
              Actions::Goto<EvolvePhaseStart>>>>>;

  static constexpr OptionString help{
      "Evolve a generalized harmonic analytic solution.\n\n"
      "The analytic solution is: Minkowski\n"
      "The numerical flux is:    UpwindFlux\n"};

  enum class Phase { Initialization, RegisterWithObserver, Evolve, Exit };

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
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
