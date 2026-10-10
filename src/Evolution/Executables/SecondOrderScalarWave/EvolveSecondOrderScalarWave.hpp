// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/SendAuxiliaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/BoundaryEvolvedVariables.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/UpdateAuxiliaryVariables.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/RandomizeVariables.hpp"
#include "ParallelAlgorithms/Actions/SpectralFilter.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeTimeStepperOrder.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

template <size_t Dim>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;

  using initial_data_list =
      SecondOrderScalarWave::Solutions::all_solutions<Dim>;

  using system = SecondOrderScalarWave::System<Dim>;
  using temporal_id = Tags::TimeStepId;

  // For labeling the yaml option for RandomizeVariables
  struct RandomizeInitialData {};

  static constexpr bool use_dg_element_collection = false;

  // The system's variables_tag is list-valued (split volume/boundary
  // variables); the DG actions act on the volume entry.
  using volume_variables_tag = ::Tags::Variables<typename system::volume_vars>;

  using analytic_solution_fields = typename system::volume_vars;
  using analytic_compute =
      evolution::Tags::AnalyticSolutionsCompute<Dim, analytic_solution_fields,
                                                false, initial_data_list>;
  using error_compute = Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;

  using observe_fields =
      tmpl::push_back<tmpl::append<typename system::volume_vars, error_tags>,
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
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, non_tensor_compute_tags>,
                       Events::time_events<system>>>>,
        tmpl::pair<evolution::BoundaryCorrection,
                   SecondOrderScalarWave::BoundaryCorrections::
                       standard_boundary_corrections<volume_dim>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::push_back<initial_data_list,
                                   evolution::initial_data::NumericData>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<SecondOrderScalarWave::BoundaryConditions::BoundaryCondition<
                       volume_dim>,
                   SecondOrderScalarWave::BoundaryConditions::
                       standard_boundary_conditions<volume_dim>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   StepChoosers::standard_slab_choosers<system>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>,
        tmpl::pair<
            Filters::Filter<volume_dim, typename system::volume_vars>,
            Filters::all_filters<volume_dim, typename system::volume_vars>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  using step_actions = tmpl::flatten<tmpl::list<
      Actions::MutateApply<
          SecondOrderScalarWave::UpdateAuxiliaryVariables<volume_dim>>,
      evolution::dg::Actions::SendAuxiliaryData<
          volume_dim, system, use_dg_element_collection, volume_variables_tag>,
      evolution::dg::Actions::ApplyAuxiliaryBoundaryCorrectionsToVariables<
          volume_dim, use_dg_element_collection, volume_variables_tag>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, use_dg_element_collection,
          volume_variables_tag>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          volume_dim, use_dg_element_collection, volume_variables_tag>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
      Actions::MutateApply<UpdateU<system>>,
      Actions::MutateApply<ChangeTimeStepperOrder<system>>,
      Actions::MutateApply<CleanHistory<system>>,
      Actions::MutateApply<evolution::dg::CleanMortarHistory<volume_dim>>,
      dg::Actions::SpectralFilter<volume_dim, typename system::volume_vars>>>;

  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          // LDG infrastructure does not suppory LTS for now.
          Initialization::TimeStepping<EvolutionMetavars, TimeStepper, false,
                                       false>,
          evolution::dg::Initialization::Domain<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
      ::Actions::RandomizeVariables<volume_variables_tag, RandomizeInitialData>,
      evolution::dg::Initialization::BoundaryEvolvedVariables<
          volume_dim, system,
          SecondOrderScalarWave::BoundaryConditions::
              standard_boundary_conditions<volume_dim>>,
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepperHistory<system>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>>,
      ::evolution::dg::Initialization::Mortars<volume_dim>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::InitializeItems<
          evolution::dg::Initialization::SpectralFilters<
              volume_dim, typename system::volume_vars>>,
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
              tmpl::flatten<
                  tmpl::list<evolution::Actions::RunEventsAndTriggers<
                                 Triggers::WhenToCheck::AtSlabs>,
                             Actions::ChangeSlabSize, step_actions,
                             Actions::MutateApply<AdvanceTime<>>,
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

  static constexpr Options::String help{
      "Evolve a second-order in space scalar save in Dim spatial dimensions "
      "using the local discontinuous Galerkin (LDG) method."};

  static constexpr auto default_phase_order = std::array<Parallel::Phase, 5>{
      Parallel::Phase::Initialization,
      Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
      Parallel::Phase::Evolve, Parallel::Phase::Exit};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
