// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmArray.hpp"
#include "AlgorithmGroup.hpp"
#include "AlgorithmNodegroup.hpp"
#include "AlgorithmSingleton.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/DomainCreators/RegisterDerivedWithCharm.cpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesGlobalTimeStepping.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/FinalTime.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/TMPL.hpp"

template <size_t Dim>
struct EvolutionMetavars {
  // Customization/"input options" to simulation
  using system = ScalarWave::System<Dim>;
  using temporal_id = Tags::TimeId;
  using analytic_solution_tag =
      CacheTags::AnalyticSolution<ScalarWave::Solutions::PlaneWave<Dim>>;
  using normal_dot_numerical_flux =
      CacheTags::NumericalFluxParams<ScalarWave::UpwindFlux<Dim>>;
  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tag_list = tmpl::list<analytic_solution_tag>;
  using domain_creator_tag = OptionTags::DomainCreator<Dim, Frame::Inertial>;

  using component_list = tmpl::list<DgElementArray<
      EvolutionMetavars,
      tmpl::list<Actions::ComputeVolumeDuDt<Dim>,
                 dg::Actions::ComputeNonconservativeBoundaryFluxes,
                 dg::Actions::SendDataForFluxes<EvolutionMetavars>,
                 dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
                 dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping,
                 Actions::UpdateU, Actions::AdvanceTime, Actions::FinalTime>>>;

  static constexpr OptionString help{
      "Evolve a Scalar Wave in Dim spatial dimension.\n\n"
      "The analytic solution is: PlaneWave\n"
      "The numerical flux is:    UpwindFlux\n"};

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
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
