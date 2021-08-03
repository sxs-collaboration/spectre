// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ApparentHorizons/Tags.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Conservative/UpdateConservatives.hpp"
#include "Evolution/Conservative/UpdatePrimitives.hpp"
#include "Evolution/DgSubcell/Actions/Initialize.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/Actions/SelectNumericalMethod.hpp"
#include "Evolution/DgSubcell/Actions/TakeTimeStep.hpp"
#include "Evolution/DgSubcell/Actions/TciAndRollback.hpp"
#include "Evolution/DgSubcell/Actions/TciAndSwitchToDg.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/ComputeBoundaryTerms.hpp"
#include "Evolution/DgSubcell/CorrectPackagedData.hpp"
#include "Evolution/DgSubcell/Events/ObserveFields.hpp"
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/GrTagsForHydro.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/SetVariablesNeededFixingToFalse.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/FixConservativesAndComputePrims.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/GrTagsForHydro.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimsAfterRollback.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ResizeAndComputePrimitives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/SwapGrTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Evolution/VariableFixing/Actions.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/Minmod.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BlastWave.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BondiHoyleAccretion.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticFieldLoop.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticRotor.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/OrszagTangVortex.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/KomissarovShock.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/MassFlux.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MemoryHelpers.hpp"
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

template <typename InitialData, typename... InterpolationTargetTags>
struct EvolutionMetavars {
  // The UseDgSubcell flag controls whether to use "standard" limiting (false)
  // or a DG-FD hybrid scheme (true).
  static constexpr bool UseDgSubcell = true;
  static constexpr size_t volume_dim = 3;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;
  using initial_data = InitialData;
  static_assert(
      evolution::is_analytic_data_v<initial_data> xor
          evolution::is_analytic_solution_v<initial_data>,
      "initial_data must be either an analytic_data or an analytic_solution");
  using equation_of_state_type = typename initial_data::equation_of_state_type;
  using system = grmhd::ValenciaDivClean::System;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;
  using initial_data_tag =
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;
  using boundary_condition_tag = initial_data_tag;
  using analytic_variables_tags =
      typename system::primitive_variables_tag::tags_list;
  using equation_of_state_tag =
      hydro::Tags::EquationOfState<equation_of_state_type>;
  // Do not limit the divergence-cleaning field Phi
  using limiter = Tags::Limiter<
      Limiters::Minmod<3, tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                     grmhd::ValenciaDivClean::Tags::TildeTau,
                                     grmhd::ValenciaDivClean::Tags::TildeS<>,
                                     grmhd::ValenciaDivClean::Tags::TildeB<>>>>;

  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using interpolator_source_vars =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
          typename InterpolationTargetTags::vars_to_interpolate_to_target...>>>;

  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;

  using ordered_list_of_primitive_recovery_schemes = tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;

  using interpolation_target_tags = tmpl::list<InterpolationTargetTags...>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                tmpl::conditional_t<
                    UseDgSubcell,
                    evolution::dg::subcell::Events::ObserveFields<
                        volume_dim, Tags::Time,
                        tmpl::append<
                            typename system::variables_tag::tags_list,
                            typename system::primitive_variables_tag::tags_list,
                            tmpl::list<
                                evolution::dg::subcell::Tags::TciStatus>>,
                        tmpl::conditional_t<
                            evolution::is_analytic_solution_v<initial_data>,
                            analytic_variables_tags, tmpl::list<>>>,
                    dg::Events::field_observations<
                        volume_dim, Tags::Time,
                        tmpl::append<typename system::variables_tag::tags_list,
                                     typename system::primitive_variables_tag::
                                         tags_list>,
                        tmpl::conditional_t<
                            evolution::is_analytic_solution_v<initial_data>,
                            analytic_variables_tags, tmpl::list<>>>>,
                Events::time_events<system>,
                intrp::Events::Interpolate<3, InterpolationTargetTags,
                                           interpolator_source_vars>...>>>,
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
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          typename InterpolationTargetTags::post_interpolation_callback...>>;

  struct SubcellOptions {
    using evolved_vars_tags = typename system::variables_tag::tags_list;
    using prim_tags = typename system::primitive_variables_tag::tags_list;
    using recons_prim_tags = tmpl::push_back<
        prim_tags,
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>;
    using fluxes_tags =
        db::wrap_tags_in<Tags::Flux, evolved_vars_tags,
                         tmpl::size_t<volume_dim>, Frame::Inertial>;

    static constexpr bool subcell_enabled = UseDgSubcell;
    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) noexcept {
      return db::get<grmhd::ValenciaDivClean::fd::Tags::Reconstructor>(box)
          .ghost_zone_size();
    }

    using DgComputeSubcellNeighborPackagedData =
        grmhd::ValenciaDivClean::subcell::NeighborPackagedData;

    using GhostDataToSlice =
        grmhd::ValenciaDivClean::subcell::PrimitiveGhostDataToSlice;
  };

  using dg_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<
                         system::primitive_from_conservative<
                             ordered_list_of_primitive_recovery_schemes>>,
                     evolution::dg::Actions::ApplyBoundaryCorrections<
                         EvolutionMetavars>>,
          tmpl::list<evolution::dg::Actions::ApplyBoundaryCorrections<
                         EvolutionMetavars>,
                     Actions::RecordTimeStepperData<>,
                     evolution::Actions::RunEventsAndDenseTriggers<
                         system::primitive_from_conservative<
                             ordered_list_of_primitive_recovery_schemes>>,
                     Actions::UpdateU<>>>,
      Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>,
      VariableFixing::Actions::FixVariables<
          grmhd::ValenciaDivClean::FixConservatives>,
      Actions::UpdatePrimitives,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives>>;

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      evolution::dg::Actions::ApplyBoundaryCorrections<EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<Actions::RecordTimeStepperData<>, Actions::UpdateU<>>>,
      // Note: The primitive variables are computed as part of the TCI.
      evolution::dg::subcell::Actions::TciAndRollback<
          grmhd::ValenciaDivClean::subcell::TciOnDgGrid<
              tmpl::front<ordered_list_of_primitive_recovery_schemes>>>,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim,
          grmhd::ValenciaDivClean::subcell::PrimitiveGhostDataOnSubcells>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      Actions::MutateApply<grmhd::ValenciaDivClean::subcell::SwapGrTags>,
      Actions::MutateApply<grmhd::ValenciaDivClean::subcell::PrimsAfterRollback<
          ordered_list_of_primitive_recovery_schemes>>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          grmhd::ValenciaDivClean::subcell::TimeDerivative>,
      Actions::RecordTimeStepperData<>, Actions::UpdateU<>,
      Actions::MutateApply<
          grmhd::ValenciaDivClean::subcell::FixConservativesAndComputePrims<
              ordered_list_of_primitive_recovery_schemes>>,
      evolution::dg::subcell::Actions::TciAndSwitchToDg<
          grmhd::ValenciaDivClean::subcell::TciOnFdGrid>,
      Actions::MutateApply<grmhd::ValenciaDivClean::subcell::SwapGrTags>,
      Actions::MutateApply<
          grmhd::ValenciaDivClean::subcell::ResizeAndComputePrims<
              ordered_list_of_primitive_recovery_schemes>>,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives,

      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<UseDgSubcell, dg_subcell_step_actions,
                          dg_step_actions>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    Register,
    LoadBalancing,
    WriteCheckpoint,
    Evolve,
    Exit
  };

  static std::string phase_name(Phase phase) noexcept {
    if (phase == Phase::LoadBalancing) {
      return "LoadBalancing";
    } else if (phase == Phase::WriteCheckpoint) {
      return "WriteCheckpoint";
    }
    ERROR(
        "Passed phase that should not be used in input file. Integer "
        "corresponding to phase is: "
        << static_cast<int>(phase));
  }

  using phase_changes =
      tmpl::list<PhaseControl::Registrars::VisitAndReturn<EvolutionMetavars,
                                                          Phase::LoadBalancing>,
                 PhaseControl::Registrars::VisitAndReturn<
                     EvolutionMetavars, Phase::WriteCheckpoint>,
                 PhaseControl::Registrars::CheckpointAndExitAfterWallclock<
                     EvolutionMetavars>>;

  using initialize_phase_change_decision_data =
      PhaseControl::InitializePhaseChangeDecisionData<phase_changes>;

  using phase_change_tags_and_combines_list =
      PhaseControl::get_phase_change_tags<phase_changes>;

  using dg_registration_list =
      tmpl::list<intrp::Actions::RegisterElementWithInterpolator,
                 observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::flatten<tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<3>,
      Initialization::Actions::GrTagsForHydro<system>,
      Initialization::Actions::ConservativeSystem<system,
                                                  equation_of_state_tag>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<3, Frame::Logical>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives,

      tmpl::conditional_t<
          UseDgSubcell,
          tmpl::list<
              evolution::dg::subcell::Actions::Initialize<
                  volume_dim, system,
                  grmhd::ValenciaDivClean::subcell::DgInitialDataTci>,
              Initialization::Actions::AddSimpleTags<
                  Initialization::subcell::GrTagsForHydro<system, volume_dim>,
                  grmhd::ValenciaDivClean::SetVariablesNeededFixingToFalse>,
              VariableFixing::Actions::FixVariables<
                  VariableFixing::FixToAtmosphere<volume_dim>>,
              Actions::MutateApply<
                  grmhd::ValenciaDivClean::subcell::SwapGrTags>,
              Actions::UpdateConservatives>,
          tmpl::list<>>,

      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data>,
          Initialization::Actions::AddComputeTags<
              tmpl::list<evolution::Tags::AnalyticCompute<
                  3, initial_data_tag, analytic_variables_tags>>>,
          tmpl::list<>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      Initialization::Actions::Minmod<3>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>>;

  using dg_element_array_component = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Phase, Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<
              Phase, Phase::Register,
              tmpl::list<intrp::Actions::RegisterElementWithInterpolator,
                         observers::Actions::RegisterEventsWithObservers,
                         Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<
                  Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                  step_actions, Actions::AdvanceTime,
                  PhaseControl::Actions::ExecutePhaseChange<phase_changes>>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, dg_element_array_component>,
        dg_registration_list, tmpl::list<>>;
  };

  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      intrp::Interpolator<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, InterpolationTargetTags>...,
      dg_element_array_component>;

  using const_global_cache_tags = tmpl::push_back<
      tmpl::conditional_t<
          UseDgSubcell,
          tmpl::list<
              grmhd::ValenciaDivClean::fd::Tags::Reconstructor,
              ::Tags::VariableFixer<grmhd::ValenciaDivClean::FixConservatives>,
              grmhd::ValenciaDivClean::subcell::Tags::TciOptions>,
          tmpl::list<>>,
      initial_data_tag,
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter,
      Tags::EventsAndTriggers,
      PhaseControl::Tags::PhaseChangeAndTriggers<phase_changes>>;

  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning.\n\n"};

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<EvolutionMetavars>&
          cache_proxy) noexcept {
    const auto next_phase = PhaseControl::arbitrate_phase_change<phase_changes>(
        phase_change_decision_data, current_phase,
        *(cache_proxy.ckLocalBranch()));
    if (next_phase.has_value()) {
      return next_phase.value();
    }
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::Register;
      case Phase::Register:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> to an integral "
            "value?");
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

struct KerrHorizon {
  using tags_to_observe =
      tmpl::list<StrahlkorperTags::EuclideanSurfaceIntegralVectorCompute<
          hydro::Tags::MassFlux<DataVector, 3>, ::Frame::Inertial>>;
  using compute_items_on_source = tmpl::list<>;
  using vars_to_interpolate_to_target =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>;
  using compute_items_on_target = tmpl::push_front<
      tags_to_observe,
      StrahlkorperTags::EuclideanAreaElementCompute<::Frame::Inertial>,
      hydro::Tags::MassFluxCompute<DataVector, 3, ::Frame::Inertial>>;
  using compute_target_points =
      intrp::TargetPoints::KerrHorizon<KerrHorizon, ::Frame::Inertial>;
  using post_interpolation_callback =
      intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, KerrHorizon,
                                                   KerrHorizon>;
  using interpolating_component =
      typename metavariables::dg_element_array_component;
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &grmhd::ValenciaDivClean::BoundaryConditions::register_derived_with_charm,
    &grmhd::ValenciaDivClean::BoundaryCorrections::register_derived_with_charm,
    &grmhd::ValenciaDivClean::fd::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
