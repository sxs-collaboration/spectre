// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/GaugeWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"
#include "Evolution/Systems/Cce/Components/KleinGordonCharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Events/ObserveFields.hpp"
#include "Evolution/Systems/Cce/Events/ObserveTimeStep.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/Triggers/TimeTriggers.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

template <template <typename> class BoundaryComponent>
struct EvolutionMetavars : CharacteristicExtractDefaults<false> {
  using system = Cce::System<evolve_ccm>;
  static constexpr bool local_time_stepping = true;
  using cce_boundary_component = BoundaryComponent<EvolutionMetavars>;
  using cce_base = CharacteristicExtractDefaults<false>;

  using evolved_swsh_tags = tmpl::append<cce_base::evolved_swsh_tags,
                                         tmpl::list<Cce::Tags::KleinGordonPsi>>;
  using evolved_swsh_dt_tags =
      tmpl::append<cce_base::evolved_swsh_dt_tags,
                   tmpl::list<Cce::Tags::KleinGordonPi>>;

  using klein_gordon_boundary_communication_tags =
      Cce::Tags::klein_gordon_worldtube_boundary_tags;

  using klein_gordon_gauge_boundary_tags = tmpl::list<
      Cce::Tags::EvolutionGaugeBoundaryValue<Cce::Tags::KleinGordonPsi>,
      Cce::Tags::EvolutionGaugeBoundaryValue<Cce::Tags::KleinGordonPi>>;

  using klein_gordon_scri_tags =
      tmpl::list<Cce::Tags::ScriPlus<Cce::Tags::KleinGordonPi>>;

  using cce_step_choosers =
      tmpl::list<StepChoosers::Constant, StepChoosers::LimitIncrease,
                 StepChoosers::ErrorControl<StepChooserUse::LtsStep,
                                            Tags::Variables<evolved_swsh_tags>,
                                            swsh_vars_selector>,
                 StepChoosers::ErrorControl<StepChooserUse::LtsStep,
                                            evolved_coordinates_variables_tag,
                                            coord_vars_selector>>;

  using klein_gordon_pre_swsh_derivative_tags =
      tmpl::list<Cce::Tags::Dy<Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>>,
                 Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>>;

  using klein_gordon_swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Cce::Tags::KleinGordonPsi,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::KleinGordonPsi,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::KleinGordonPsi,
                                       Spectral::Swsh::Tags::EthEth>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::KleinGordonPsi,
                                       Spectral::Swsh::Tags::EthEthbar>>;
  using klein_gordon_transform_buffer_tags = tmpl::list<
      Spectral::Swsh::Tags::SwshTransform<Cce::Tags::KleinGordonPsi>,
      Spectral::Swsh::Tags::SwshTransform<
          Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::KleinGordonPsi, Spectral::Swsh::Tags::Eth>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::KleinGordonPsi, Spectral::Swsh::Tags::Ethbar>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>, Spectral::Swsh::Tags::Eth>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::Dy<Cce::Tags::KleinGordonPsi>,
          Spectral::Swsh::Tags::Ethbar>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::KleinGordonPsi, Spectral::Swsh::Tags::EthEth>>,
      Spectral::Swsh::Tags::SwshTransform<Spectral::Swsh::Tags::Derivative<
          Cce::Tags::KleinGordonPsi, Spectral::Swsh::Tags::EthEthbar>>>;

  using klein_gordon_source_tags = tmpl::flatten<
      tmpl::transform<Cce::bondi_hypersurface_step_tags,
                      tmpl::bind<Cce::Tags::KleinGordonSource, tmpl::_1>>>;

  using klein_gordon_cce_integrand_tags =
      tmpl::list<Cce::Tags::PoleOfIntegrand<Cce::Tags::KleinGordonPi>,
                 Cce::Tags::RegularIntegrand<Cce::Tags::KleinGordonPi>>;

  using scri_values_to_observe =
      tmpl::append<cce_base::scri_values_to_observe,
                   tmpl::list<Cce::Tags::ScriPlus<Cce::Tags::KleinGordonPsi>>>;

  using cce_scri_tags =
      tmpl::append<cce_base::cce_scri_tags,
                   tmpl::list<Cce::Tags::ScriPlus<Cce::Tags::KleinGordonPsi>>>;

  using component_list =
      tmpl::list<observers::ObserverWriter<EvolutionMetavars>,
                 cce_boundary_component,
                 Cce::KleinGordonCharacteristicEvolution<EvolutionMetavars>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, cce_step_choosers>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   StepChoosers::standard_slab_choosers<
                       system, local_time_stepping, false>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<Event, tmpl::list<Cce::Events::ObserveFields,
                                     Cce::Events::ObserveTimeStep>>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  using observed_reduction_data_tags = tmpl::list<>;

  static constexpr Options::String help{
      "Perform Cauchy Characteristic Extraction for the Klein-Gordon system "
      "coupled with General Relativity, using .h5 input data."};

  static constexpr std::array<Parallel::Phase, 4> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Evolve,
       Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
