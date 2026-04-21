// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/DiscontinuousGalerkin/BackgroundGrVars.hpp"
#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/ChangeFixedLtsRatio.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/FixedLtsRatio.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/NonconformingEqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SetupEqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.hpp"
#include "Evolution/Imex/Actions/DoImplicitStep.hpp"
#include "Evolution/Imex/Actions/RecordTimeStepperData.hpp"
#include "Evolution/Imex/CleanHistory.hpp"
#include "Evolution/Imex/ImplicitDenseOutput.hpp"
#include "Evolution/Imex/Initialize.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Initialize.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1HydroCoupling.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/SpectralFilter.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/ChangeFixedLtsRatio.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/RadiationTransport/M1Grey/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/RadiationTransport/M1Grey/Factory.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeTimeStepperOrder.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  static constexpr size_t volume_dim = 3;

  // Set list of neutrino species to be used by M1 code
  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>>;

  using system = RadiationTransport::M1Grey::System<neutrino_species>;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = ImexTimeStepper;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  using initial_data_list =
      tmpl::append<RadiationTransport::M1Grey::AnalyticData::all_data,
                   RadiationTransport::M1Grey::Solutions::all_solutions>;

  using analytic_variables_tags = typename system::variables_tag::tags_list;

  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_variables_tags, false, initial_data_list>;
  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;
  using observe_fields = tmpl::push_back<
      tmpl::append<typename system::variables_tag::tags_list,
                   typename system::primitive_variables_tag::tags_list,
                   error_tags>,
      domain::Tags::Coordinates<volume_dim, Frame::Grid>,
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
  using non_tensor_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>,
                 analytic_compute, error_compute>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                dg::Events::field_observations<volume_dim, observe_fields,
                                               non_tensor_compute_tags>,
                Events::time_events<system>,
                tmpl::conditional_t<local_time_stepping,
                                    dg::Events::ChangeFixedLtsRatio<volume_dim>,
                                    tmpl::list<>>>>>,
        tmpl::pair<evolution::BoundaryCorrection,
                   RadiationTransport::M1Grey::BoundaryCorrections::
                       standard_boundary_corrections<neutrino_species>>,
        tmpl::pair<ImexTimeStepper, TimeSteppers::imex_time_steppers>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<RadiationTransport::M1Grey::BoundaryConditions::
                       BoundaryCondition<neutrino_species>,
                   RadiationTransport::M1Grey::BoundaryConditions::
                       standard_boundary_conditions<neutrino_species>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system, false>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::append<StepChoosers::standard_slab_choosers<
                                    system, local_time_stepping, false>,
                                tmpl::conditional_t<
                                    local_time_stepping,
                                    tmpl::list<evolution::dg::StepChoosers::
                                                   FixedLtsRatio<volume_dim>>,
                                    tmpl::list<>>>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
        tmpl::pair<Filters::Filter<volume_dim,
                                   typename system::variables_tag::tags_list>,
                   Filters::all_filters<
                       volume_dim, typename system::variables_tag::tags_list>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  static_assert(not local_time_stepping);
  using step_actions = tmpl::flatten<tmpl::list<
      Actions::MutateApply<
          evolution::dg::BackgroundGrVars<system, EvolutionMetavars>>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      imex::Actions::RecordTimeStepperData<system>,
      evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
          evolution::dg::ApplyLtsDenseBoundaryCorrections<EvolutionMetavars>,
          imex::ImplicitDenseOutput<system>>>,
      Actions::MutateApply<UpdateU<system, local_time_stepping>>,
      evolution::dg::Actions::ApplyLtsBoundaryCorrections<
          volume_dim, use_dg_element_collection>,
      imex::Actions::DoImplicitStep<system>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<Actions::MutateApply<ChangeTimeStepperOrder<system>>>,
          tmpl::list<>>,
      Actions::MutateApply<CleanHistory<system>>,
      Actions::MutateApply<imex::CleanHistory<system>>,
      Actions::MutateApply<evolution::dg::CleanMortarHistory<volume_dim>>,
      dg::Actions::SpectralFilter,
      Actions::MutateApply<typename RadiationTransport::M1Grey::
                               ComputeM1Closure<neutrino_species>>>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using equal_rate_regions =
      tmpl::list<evolution::dg::NonconformingEqualRateRegions<volume_dim>>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase,
                                       false>,
          evolution::dg::Initialization::Domain<EvolutionMetavars>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::AddSimpleTags<
          evolution::dg::BackgroundGrVars<system, EvolutionMetavars>>,
      Initialization::Actions::ConservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<volume_dim, Frame::ElementLogical>>,
      RadiationTransport::M1Grey::Actions::InitializeM1Tags<system>,
      Initialization::Actions::InitializeItems<imex::Initialize<system>>,
      Actions::MutateApply<typename RadiationTransport::M1Grey::
                               ComputeM1Closure<neutrino_species>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars,
                                                  local_time_stepping>>,
      ::evolution::dg::Initialization::Mortars<volume_dim>,
      evolution::dg::Initialization::Actions::SetupEqualRateRegions<
          EvolutionMetavars, volume_dim, equal_rate_regions>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::InitializeItems<
          evolution::dg::Initialization::SpectralFilters<
              volume_dim, typename system::variables_tag::tags_list>>,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<Parallel::Phase::Restart,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::WriteCheckpoint,
              tmpl::list<evolution::Actions::RunEventsAndTriggers<
                             Triggers::WhenToCheck::AtCheckpoints>,
                         Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::flatten<tmpl::list<
                  std::conditional_t<local_time_stepping,
                                     evolution::Actions::RunEventsAndTriggers<
                                         Triggers::WhenToCheck::AtSteps>,
                                     tmpl::list<>>,
                  evolution::Actions::RunEventsAndTriggers<
                      Triggers::WhenToCheck::AtSlabs>,
                  Actions::ChangeSlabSize,
                  std::conditional_t<
                      local_time_stepping,
                      evolution::dg::Actions::ChangeFixedLtsRatio,
                      tmpl::list<>>,
                  step_actions, Actions::MutateApply<AdvanceTime<>>,
                  PhaseControl::Actions::ExecutePhaseChange>>>>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, dg_registration_list>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array>;

  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData>;

  static constexpr Options::String help{"Evolve the M1Grey system. \n\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
