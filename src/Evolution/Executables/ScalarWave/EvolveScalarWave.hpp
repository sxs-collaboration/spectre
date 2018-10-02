// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/DomainCreators/RegisterDerivedWithCharm.cpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/Actions.hpp"
#include "Evolution/Systems/ScalarWave/Equations.hpp"  // IWYU pragma: keep // for UpwindFlux
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesGlobalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/FinalTime.hpp"              // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
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
  static constexpr bool local_time_stepping = false;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<ScalarWave::Solutions::PlaneWave<Dim>>;
  using normal_dot_numerical_flux =
      OptionTags::NumericalFluxParams<ScalarWave::UpwindFlux<Dim>>;
  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tag_list = tmpl::list<analytic_solution_tag>;
  using domain_creator_tag = OptionTags::DomainCreator<Dim, Frame::Inertial>;

  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      DgElementArray<
          EvolutionMetavars,
          tmpl::list<Actions::AdvanceTime, ScalarWave::Actions::Observe,
                     Actions::FinalTime, Actions::ComputeVolumeDuDt,
                     dg::Actions::ComputeNonconservativeBoundaryFluxes,
                     dg::Actions::SendDataForFluxes<EvolutionMetavars>,
                     dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
                     dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping,
                     Actions::RecordTimeStepperData, Actions::UpdateU>>>;

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
    &setup_error_handling, &DomainCreators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
