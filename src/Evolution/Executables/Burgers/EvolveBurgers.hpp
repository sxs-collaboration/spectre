// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/Burgers/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Burgers/Subcell/GhostData.hpp"
#include "Evolution/Systems/Burgers/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/Burgers/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/Burgers/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/Burgers/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/Burgers/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Burgers/Sinusoid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Bump.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/ChangeSlabSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"       // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

template <typename InitialData>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = 1;
  using system = Burgers::System;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  // The use_dg_subcell flag controls whether to use "standard" limiting (false)
  // or a DG-FD hybrid scheme (true).
  static constexpr bool use_dg_subcell = true;

  using initial_data = InitialData;
  static_assert(
      is_analytic_data_v<initial_data> xor is_analytic_solution_v<initial_data>,
      "initial_data must be either an analytic_data or an analytic_solution");
  using initial_data_tag =
      tmpl::conditional_t<is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;

  using limiter =
      Tags::Limiter<Limiters::Minmod<1, system::variables_tag::tags_list>>;

  using analytic_variables_tags = typename system::variables_tag::tags_list;
  using analytic_compute =
      evolution::Tags::AnalyticSolutionsCompute<volume_dim,
                                                analytic_variables_tags>;
  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;
  using observe_fields = tmpl::push_back<
      tmpl::append<typename system::variables_tag::tags_list, error_tags,
                   tmpl::conditional_t<
                       use_dg_subcell,
                       tmpl::list<evolution::dg::subcell::Tags::TciStatus>,
                       tmpl::list<>>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<volume_dim,
                                                                   Frame::Grid>,
          domain::Tags::Coordinates<volume_dim, Frame::Grid>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::Inertial>,
          domain::Tags::Coordinates<volume_dim, Frame::Inertial>>>;
  using non_tensor_compute_tags = tmpl::list<
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverMeshCompute<volume_dim>,
          ::Events::Tags::ObserverMeshCompute<volume_dim>>,
      analytic_compute, error_compute>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Burgers::BoundaryConditions::BoundaryCondition,
                   Burgers::BoundaryConditions::standard_boundary_conditions>,
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<Event, tmpl::flatten<tmpl::list<
                              Events::Completion,
                              dg::Events::field_observations<
                                  volume_dim, Tags::Time, observe_fields,
                                  non_tensor_compute_tags>,
                              Events::time_events<system>>>>,
        tmpl::pair<PhaseChange,
                   tmpl::list<PhaseControl::VisitAndReturn<
                                  Parallel::Phase::LoadBalancing>,
                              PhaseControl::CheckpointAndExitAfterWallclock>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            StepChoosers::standard_slab_choosers<system, local_time_stepping>>,
        tmpl::pair<StepController, StepControllers::standard_step_controllers>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  struct SubcellOptions {
    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = true;

    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<Burgers::fd::Tags::Reconstructor>(box).ghost_zone_size();
    }

    using DgComputeSubcellNeighborPackagedData =
        Burgers::subcell::NeighborPackagedData;

    using GhostDataToSlice = Burgers::subcell::GhostDataToSlice;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>>>>;

  using dg_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<volume_dim, system>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<
                         tmpl::list<evolution::dg::ApplyBoundaryCorrections<
                             EvolutionMetavars, true>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         EvolutionMetavars>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  EvolutionMetavars>,
              Actions::RecordTimeStepperData<>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<>>>,
      Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>>>;

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,
      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      evolution::dg::Actions::ComputeTimeDerivative<volume_dim, system>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<
              Actions::RecordTimeStepperData<>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<>>>,
      evolution::dg::subcell::Actions::TciAndRollback<
          Burgers::subcell::TciOnDgGrid>,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,
      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim, Burgers::subcell::GhostDataOnSubcells>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          Burgers::subcell::TimeDerivative>,
      Actions::RecordTimeStepperData<>,
      evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
      Actions::UpdateU<>,
      evolution::dg::subcell::Actions::TciAndSwitchToDg<
          Burgers::subcell::TciOnFdGrid>,
      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using const_global_cache_tags = tmpl::list<
      initial_data_tag,
      tmpl::conditional_t<use_dg_subcell,
                          tmpl::list<Burgers::fd::Tags::Reconstructor>,
                          tmpl::list<>>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<1>,
      Initialization::Actions::ConservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<1, Frame::ElementLogical>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,

      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::Initialize<
                  volume_dim, system, Burgers::subcell::DgInitialDataTci>,
              Actions::MutateApply<Burgers::subcell::SetInitialRdmpData>>,
          tmpl::list<>>,

      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      Initialization::Actions::Minmod<1>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type =
        std::conditional_t<std::is_same_v<ParallelComponent, dg_element_array>,
                           dg_registration_list, tmpl::list<>>;
  };

  using component_list =
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 dg_element_array>;

  static constexpr Options::String help{
      "Evolve the Burgers equation.\n\n"
      "The analytic solution is: Linear\n"
      "The numerical flux is:    LocalLaxFriedrichs\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &Burgers::BoundaryCorrections::register_derived_with_charm,
    &Burgers::fd::register_derived_with_charm,
    &Parallel::register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
