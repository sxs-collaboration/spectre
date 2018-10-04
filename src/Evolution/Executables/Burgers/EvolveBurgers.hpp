// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// IWYU pragma: no_include <pup.h>
#include <vector>

#include "Domain/DomainCreators/RegisterDerivedWithCharm.cpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"  // IWYU pragma: keep
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/InitializeElement.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Tags.hpp"
#include "Evolution/Systems/Burgers/Equations.hpp"  // IWYU pragma: keep // for LocalLaxFriedrichsFlux
#include "Evolution/Systems/Burgers/System.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesGlobalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/GotoAction.hpp"  // IWYU pragma: keep
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"  // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"  // IWYU pragma: keep
#include "Time/Actions/FinalTime.hpp"  // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"  // IWYU pragma: keep
#include "Time/StepChoosers/Cfl.hpp"  // IWYU pragma: keep
#include "Time/StepChoosers/Constant.hpp"  // IWYU pragma: keep
#include "Time/StepChoosers/Increase.hpp"  // IWYU pragma: keep
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/TMPL.hpp"

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
      OptionTags::NumericalFluxParams<Burgers::LocalLaxFriedrichsFlux>;
  using limiter = OptionTags::SlopeLimiterParams<
      SlopeLimiters::Minmod<1, system::variables_tag::tags_list>>;
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>>;
  using domain_creator_tag = OptionTags::DomainCreator<1, Frame::Inertial>;

  using step_choosers =
      tmpl::list<StepChoosers::Register::Cfl<1, Frame::Inertial>,
                 StepChoosers::Register::Constant,
                 StepChoosers::Register::Increase>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      Actions::ComputeVolumeFluxes,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeVolumeDuDt,
      dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping>,
      Actions::RecordTimeStepperData>>;
  using update_variables = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      Actions::UpdateU,
      SlopeLimiters::Actions::SendData<EvolutionMetavars>,
      SlopeLimiters::Actions::Limit<EvolutionMetavars>>>;

  struct EvolvePhaseStart;
  using component_list = tmpl::list<DgElementArray<
      EvolutionMetavars, dg::Actions::InitializeElement<1>,
      tmpl::flatten<tmpl::list<
          SelfStart::self_start_procedure<compute_rhs, update_variables>,
          Actions::Label<EvolvePhaseStart>, Actions::AdvanceTime,
          Actions::FinalTime,
          tmpl::conditional_t<local_time_stepping,
                              Actions::ChangeStepSize<step_choosers>,
                              tmpl::list<>>,
          compute_rhs, update_variables, Actions::Goto<EvolvePhaseStart>>>>>;

  static constexpr OptionString help{
      "Evolve the Burgers equation.\n\n"
      "The analytic solution is: Linear\n"
      "The numerical flux is:    LocalLaxFriedrichs\n"};

  enum class Phase {
    Initialization,
    Evolve,
    Exit
  };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Evolve : Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &DomainCreators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
