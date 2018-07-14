// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Domain/DomainCreators/RegisterDerivedWithCharm.cpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"  // IWYU pragma: keep
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/Burgers/Equations.hpp"  // IWYU pragma: keep // for LocalLaxFriedrichsFlux
#include "Evolution/Systems/Burgers/System.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesGlobalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"  // IWYU pragma: keep
#include "Time/Actions/FinalTime.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"  // IWYU pragma: keep
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
  using analytic_solution_tag =
      CacheTags::AnalyticSolution<Burgers::Solutions::Linear>;
  using normal_dot_numerical_flux =
      CacheTags::NumericalFluxParams<Burgers::LocalLaxFriedrichsFlux>;
  using const_global_cache_tag_list = tmpl::list<analytic_solution_tag>;
  using domain_creator_tag = OptionTags::DomainCreator<1, Frame::Inertial>;

  using component_list = tmpl::list<DgElementArray<
      EvolutionMetavars,
      tmpl::list<Actions::ComputeVolumeFluxes,
                 dg::Actions::SendDataForFluxes<EvolutionMetavars>,
                 Actions::ComputeVolumeDuDt<1>,
                 dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
                 dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping,
                 Actions::UpdateU, Actions::AdvanceTime, Actions::FinalTime>>>;

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
