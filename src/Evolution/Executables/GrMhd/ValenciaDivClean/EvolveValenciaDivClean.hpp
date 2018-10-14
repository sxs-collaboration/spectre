// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Domain/DomainCreators/RegisterDerivedWithCharm.cpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"
#include "Evolution/Actions/ComputeVolumeSources.hpp"
#include "Evolution/Conservative/UpdatePrimitives.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Initialize.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Observe.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesGlobalTimeStepping.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/GotoAction.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/FinalTime.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Functional.hpp"
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
  using system = grmhd::ValenciaDivClean::System;
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = false;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<grmhd::Solutions::SmoothFlow>;
  using analytic_variables_tags =
      typename system::primitive_variables_tag::tags_list;
  using equation_of_state_tag = hydro::Tags::EquationOfState<
      typename analytic_solution_tag::type::equation_of_state_type>;
  using normal_dot_numerical_flux = OptionTags::NumericalFluxParams<
      dg::NumericalFluxes::LocalLaxFriedrichs<system>>;
  using limiter = OptionTags::SlopeLimiterParams<
      SlopeLimiters::Minmod<3, system::variables_tag::tags_list>>;

  using step_choosers =
      tmpl::list<StepChoosers::Register::Cfl<3, Frame::Inertial>,
                 StepChoosers::Register::Constant,
                 StepChoosers::Register::Increase>;

  // hack this has to be synchronized with the Observe action :(
  using Redum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                         funcl::Sqrt<funcl::Divides<>>,
                                         std::index_sequence<1>>;
  using reduction_data_tags = tmpl::list<observers::Tags::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>, Redum, Redum, Redum>>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      Actions::ComputeVolumeFluxes,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeVolumeSources, Actions::ComputeVolumeDuDt,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping>,
      Actions::RecordTimeStepperData>>;

  using update_variables = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      Actions::UpdateU, SlopeLimiters::Actions::SendData<EvolutionMetavars>,
      SlopeLimiters::Actions::Limit<EvolutionMetavars>,
      Actions::UpdatePrimitives>>;

  struct EvolvePhaseStart;
  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      DgElementArray<
          EvolutionMetavars, grmhd::ValenciaDivClean::Actions::Initialize<3>,
          tmpl::flatten<tmpl::list<
              SelfStart::self_start_procedure<compute_rhs, update_variables>,
              Actions::Label<EvolvePhaseStart>, Actions::AdvanceTime,
              grmhd::ValenciaDivClean::Actions::Observe, Actions::FinalTime,
              tmpl::conditional_t<local_time_stepping,
                                  Actions::ChangeStepSize<step_choosers>,
                                  tmpl::list<>>,
              compute_rhs, update_variables,
              Actions::Goto<EvolvePhaseStart>>>>>;

  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>,
                 OptionTags::DampingParameter>;

  using domain_creator_tag = OptionTags::DomainCreator<3, Frame::Inertial>;

  static constexpr OptionString help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning.\n\n"};

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
    &setup_error_handling, &DomainCreators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<EvolutionMetavars::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
