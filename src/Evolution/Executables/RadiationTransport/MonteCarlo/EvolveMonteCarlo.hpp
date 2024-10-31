// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/BackgroundGrVars.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/DiscontinuousGalerkin/BackgroundGrVars.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Particles/MonteCarlo/Actions/InitializeMonteCarlo.hpp"
#include "Evolution/Particles/MonteCarlo/Actions/TimeStepActions.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunication.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationTags.hpp"
#include "Evolution/Particles/MonteCarlo/System.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/SwapGrTags.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
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
#include "ParallelAlgorithms/Actions/LimiterActions.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/InitialMagneticField.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/LowerSpatialFourVelocity.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
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

// NEED:
// Initial data

template <size_t EnergyBins, size_t NeutrinoSpecies>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = 3;

  using system = Particles::MonteCarlo::System;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = TimeStepper;
  static constexpr bool use_dg_subcell = true;

  using initial_data_list =
      grmhd::ValenciaDivClean::InitialData::initial_data_list;
  using equation_of_state_tag = hydro::Tags::GrmhdEquationOfState;

  struct SubcellOptions {
    using evolved_vars_tags = typename system::variables_tag::tags_list;

    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = true;
  };

  using observe_fields = tmpl::list<
      // tmpl::push_back<
      // tmpl::append<typename system::mc_variables_tag::tags_list>,
      domain::Tags::Coordinates<volume_dim, Frame::Grid>,
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
  using non_tensor_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>>;

  using analytic_variables_tags = typename system::variables_tag::tags_list;
  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_variables_tags, false, initial_data_list>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                dg::Events::field_observations<
                    volume_dim, observe_fields,
                    non_tensor_compute_tags>  //,
                                              // Events::time_events<system>
                >>>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
        tmpl::pair<
            grmhd::AnalyticData::InitialMagneticFields::InitialMagneticField,
            grmhd::AnalyticData::InitialMagneticFields::
                initial_magnetic_fields>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   StepChoosers::standard_slab_choosers<system, false, false>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  using step_actions = tmpl::flatten<tmpl::list<
      Particles::MonteCarlo::Actions::SendDataForMcCommunication<
          volume_dim,
          // No local time stepping
          false, Particles::MonteCarlo::CommunicationStep::PreStep>,
      Particles::MonteCarlo::Actions::ReceiveDataForMcCommunication<
          volume_dim, Particles::MonteCarlo::CommunicationStep::PreStep>,
      Particles::MonteCarlo::Actions::TakeTimeStep<EnergyBins, NeutrinoSpecies>,
      Particles::MonteCarlo::Actions::SendDataForMcCommunication<
          volume_dim,
          // No local time stepping
          false, Particles::MonteCarlo::CommunicationStep::PostStep>,
      Particles::MonteCarlo::Actions::ReceiveDataForMcCommunication<
          volume_dim, Particles::MonteCarlo::CommunicationStep::PostStep>,
      // Actions::RecordTimeStepperData<system>,
      // evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
      Actions::CleanHistory<system, false>>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<volume_dim>>,
      Initialization::Actions::AddSimpleTags<
          evolution::dg::BackgroundGrVars<system, EvolutionMetavars, true>>,
      evolution::dg::subcell::Actions::SetSubcellGrid<volume_dim, system,
                                                      false>,
      Initialization::Actions::AddSimpleTags<
          evolution::dg::subcell::BackgroundGrVars<system, EvolutionMetavars,
                                                   true, false>>,
      Actions::MutateApply<grmhd::ValenciaDivClean::subcell::SwapGrTags>,
      Initialization::Actions::InitializeMCTags<system, EnergyBins,
                                                NeutrinoSpecies>,
      Initialization::Actions::AddComputeTags<tmpl::list<
          hydro::Tags::LowerSpatialFourVelocityCompute,
          Particles::MonteCarlo::InverseJacobianInertialToFluidCompute,
          domain::Tags::JacobianCompute<4, Frame::Inertial, Frame::Fluid>>>,
      // Initialization::TimeStepperHistory<EvolutionMetavars>
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          //,
          // Parallel::PhaseActions<
          //     Parallel::Phase::InitializeTimeStepperHistory,
          //     SelfStart::self_start_procedure<step_actions, system>>,

          //  Parallel::PhaseActions<Parallel::Phase::Register,
          //                         tmpl::list<dg_registration_list,
          //                 Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<Particles::MonteCarlo::Actions::TakeTimeStep<
                             EnergyBins, NeutrinoSpecies>,
                         Parallel::Actions::TerminatePhase>>
          //      tmpl::list<evolution::Actions::RunEventsAndTriggers,
          //                 Actions::ChangeSlabSize, //step_actions,
          //                 Actions::AdvanceTime,
          //                 PhaseControl::Actions::ExecutePhaseChange>>
          >>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, dg_registration_list>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array>;

  using const_global_cache_tags = tmpl::list<
      equation_of_state_tag, evolution::initial_data::Tags::InitialData,
      Particles::MonteCarlo::Tags::InteractionRatesTable<EnergyBins,
                                                         NeutrinoSpecies>>;

  static constexpr Options::String help{
      "Evolve Monte Carlo transport (without coupling to hydro).\n\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
