// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/GaugeWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/RegisterInitializeJWithCharm.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

template <template <typename> class BoundaryComponent>
struct EvolutionMetavars {
  using system = Cce::System;
  static constexpr bool local_time_stepping = true;

  static constexpr bool uses_partially_flat_cartesian_coordinates = false;

  using evolved_swsh_tag = Cce::Tags::BondiJ;
  using evolved_swsh_dt_tag = Cce::Tags::BondiH;
  using evolved_coordinates_variables_tag = Tags::Variables<
      std::conditional_t<uses_partially_flat_cartesian_coordinates,
                         tmpl::list<Cce::Tags::CauchyCartesianCoords,
                                    Cce::Tags::InertialCartesianCoords,
                                    Cce::Tags::InertialRetardedTime>,
                         tmpl::list<Cce::Tags::CauchyCartesianCoords,
                                    Cce::Tags::InertialRetardedTime>>>;

  struct swsh_vars_selector {
    static std::string name() noexcept { return "SwshVars"; }
  };

  struct coord_vars_selector {
    static std::string name() noexcept { return "CoordVars"; }
  };

  using cce_boundary_communication_tags =
      Cce::Tags::characteristic_worldtube_boundary_tags<
          Cce::Tags::BoundaryValue>;

  using cce_gauge_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::transform<
          tmpl::list<Cce::Tags::BondiR, Cce::Tags::DuRDividedByR,
                     Cce::Tags::BondiJ, Cce::Tags::Dr<Cce::Tags::BondiJ>,
                     Cce::Tags::BondiBeta, Cce::Tags::BondiQ, Cce::Tags::BondiU,
                     Cce::Tags::BondiW, Cce::Tags::BondiH>,
          tmpl::bind<Cce::Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Cce::Tags::BondiUAtScri, Cce::Tags::GaugeC, Cce::Tags::GaugeD,
      Cce::Tags::GaugeOmega, Cce::Tags::Du<Cce::Tags::GaugeOmega>,
      std::conditional_t<
          uses_partially_flat_cartesian_coordinates,
          tmpl::list<
              Cce::Tags::CauchyGaugeC, Cce::Tags::CauchyGaugeD,
              Cce::Tags::CauchyGaugeOmega,
              Spectral::Swsh::Tags::Derivative<Cce::Tags::CauchyGaugeOmega,
                                               Spectral::Swsh::Tags::Eth>>,
          tmpl::list<>>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Cce::all_boundary_pre_swsh_derivative_tags_for_scri,
      Cce::all_boundary_swsh_derivative_tags_for_scri>>;

  using scri_values_to_observe =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::Du<Cce::Tags::TimeIntegral<
                     Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>,
                 Cce::Tags::EthInertialRetardedTime>;

  using cce_scri_tags =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>,
                 Cce::Tags::EthInertialRetardedTime>;
  using cce_integrand_tags = tmpl::flatten<tmpl::transform<
      Cce::bondi_hypersurface_step_tags,
      tmpl::bind<Cce::integrand_terms_to_compute_for_bondi_variable,
                 tmpl::_1>>>;
  using cce_integration_independent_tags =
      tmpl::push_back<Cce::pre_computation_tags, Cce::Tags::DuRDividedByR>;
  using cce_temporary_equations_tags = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::transform<cce_integrand_tags,
                      tmpl::bind<Cce::integrand_temporary_tags, tmpl::_1>>>>;
  using cce_pre_swsh_derivatives_tags = Cce::all_pre_swsh_derivative_tags;
  using cce_transform_buffer_tags = Cce::all_transform_buffer_tags;
  using cce_swsh_derivative_tags = Cce::all_swsh_derivative_tags;
  using cce_angular_coordinate_tags =
      std::conditional_t<uses_partially_flat_cartesian_coordinates,
                         tmpl::list<Cce::Tags::CauchyAngularCoords,
                                    Cce::Tags::PartiallyFlatAngularCoords>,
                         tmpl::list<Cce::Tags::CauchyAngularCoords>>;

  using cce_boundary_component = BoundaryComponent<EvolutionMetavars>;

  using component_list =
      tmpl::list<observers::ObserverWriter<EvolutionMetavars>,
                 cce_boundary_component,
                 Cce::CharacteristicEvolution<EvolutionMetavars>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::list<StepChoosers::Constant<StepChooserUse::LtsStep>,
                              StepChoosers::Increase<StepChooserUse::LtsStep>,
                              StepChoosers::ErrorControl<
                                  Tags::Variables<tmpl::list<evolved_swsh_tag>>,
                                  swsh_vars_selector>,
                              StepChoosers::ErrorControl<
                                  evolved_coordinates_variables_tag,
                                  coord_vars_selector>>>,
        tmpl::pair<StepController, StepControllers::standard_step_controllers>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>>;
  };

  using observed_reduction_data_tags = tmpl::list<>;

  static constexpr Options::String help{
      "Perform Cauchy Characteristic Extraction using .h5 input data.\n"
      "Uses regularity-preserving formulation."};

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    Evolve,
    Exit
  };

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<
          tuples::TaggedTuple<Tags...>*> /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    if (current_phase == Phase::Initialization) {
      return Phase::InitializeTimeStepperHistory;
    } else if (current_phase == Phase::InitializeTimeStepperHistory) {
      return Phase::Evolve;
    } else {
      return Phase::Exit;
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &Cce::register_initialize_j_with_charm<
        metavariables::uses_partially_flat_cartesian_coordinates>,
    &Parallel::register_derived_classes_with_charm<
        Cce::WorldtubeBufferUpdater<Cce::cce_metric_input_tags>>,
    &Parallel::register_derived_classes_with_charm<
        Cce::WorldtubeBufferUpdater<Cce::cce_bondi_input_tags>>,
    &Parallel::register_derived_classes_with_charm<Cce::WorldtubeDataManager>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<intrp::SpanInterpolator>,
    &Parallel::register_derived_classes_with_charm<
        Cce::Solutions::WorldtubeData>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
