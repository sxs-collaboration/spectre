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
#include "Evolution/DgSubcell/NeighborReconstructedFaceSolution.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/PrepareNeighborData.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Krivodonova.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/GrTagsForHydro.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Flattener.hpp"
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
#include "Evolution/VariableFixing/Actions.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/FiniteDifference/Minmod.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/KerrHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/SpecifiedPoints.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BlastWave.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BondiHoyleAccretion.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/KhInstability.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticFieldLoop.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticRotor.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/OrszagTangVortex.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/KomissarovShock.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/RegisterDerivedWithCharm.hpp"
#include "PointwiseFunctions/Hydro/MassFlux.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"
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
  // The use_dg_subcell flag controls whether to use "standard" limiting (false)
  // or a DG-FD hybrid scheme (true).
  static constexpr bool use_dg_subcell = true;
  static constexpr size_t volume_dim = 3;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;
  using initial_data = InitialData;
  static_assert(
      is_analytic_data_v<initial_data> xor is_analytic_solution_v<initial_data>,
      "initial_data must be either an analytic_data or an analytic_solution");
  using eos_base = typename EquationsOfState::get_eos_base<
      typename initial_data::equation_of_state_type>;
  using equation_of_state_type = typename std::unique_ptr<eos_base>;
  using system = grmhd::ValenciaDivClean::System;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;
  using initial_data_tag =
      tmpl::conditional_t<is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;
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

  using interpolator_source_vars =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
          typename InterpolationTargetTags::vars_to_interpolate_to_target...>>>;

  using ordered_list_of_primitive_recovery_schemes = tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;

  using interpolation_target_tags = tmpl::list<InterpolationTargetTags...>;

  using analytic_compute =
      evolution::Tags::AnalyticSolutionsCompute<volume_dim,
                                                analytic_variables_tags>;
  using error_compute = Tags::ErrorsCompute<analytic_variables_tags>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_variables_tags>;
  using observe_fields = tmpl::push_back<
      tmpl::append<typename system::variables_tag::tags_list,
                   typename system::primitive_variables_tag::tags_list,
                   error_tags,
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
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<Event, tmpl::flatten<tmpl::list<
                              Events::Completion,
                              dg::Events::field_observations<
                                  volume_dim, Tags::Time, observe_fields,
                                  non_tensor_compute_tags>,
                              Events::time_events<system>,
                              intrp::Events::InterpolateWithoutInterpComponent<
                                  3, InterpolationTargetTags, EvolutionMetavars,
                                  interpolator_source_vars>...>>>,
        tmpl::pair<
            grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition,
            grmhd::ValenciaDivClean::BoundaryConditions::
                standard_boundary_conditions>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                PhaseControl::VisitAndReturn<Parallel::Phase::LoadBalancing>,
                PhaseControl::VisitAndReturn<Parallel::Phase::WriteCheckpoint>,
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

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  struct SubcellOptions {
    using evolved_vars_tags = typename system::variables_tag::tags_list;
    using prim_tags = typename system::primitive_variables_tag::tags_list;
    using recons_prim_tags = tmpl::push_back<
        prim_tags,
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>;
    using fluxes_tags =
        db::wrap_tags_in<Tags::Flux, evolved_vars_tags,
                         tmpl::size_t<volume_dim>, Frame::Inertial>;

    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = true;

    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<grmhd::ValenciaDivClean::fd::Tags::Reconstructor>(box)
          .ghost_zone_size();
    }

    using DgComputeSubcellNeighborPackagedData =
        grmhd::ValenciaDivClean::subcell::NeighborPackagedData;

    using GhostDataToSlice =
        grmhd::ValenciaDivClean::subcell::PrimitiveGhostDataToSlice;
  };

  using events_and_dense_triggers_subcell_postprocessors =
      tmpl::list<AlwaysReadyPostprocessor<
          grmhd::ValenciaDivClean::subcell::FixConservativesAndComputePrims<
              ordered_list_of_primitive_recovery_schemes>>>;

  using dg_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<volume_dim, system,
                                                    AllStepChoosers>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                         evolution::dg::ApplyBoundaryCorrections<
                             EvolutionMetavars, true>,
                         system::primitive_from_conservative<
                             ordered_list_of_primitive_recovery_schemes>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         EvolutionMetavars>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  EvolutionMetavars>,
              Actions::RecordTimeStepperData<>,
              evolution::Actions::RunEventsAndDenseTriggers<
                  tmpl::list<system::primitive_from_conservative<
                      ordered_list_of_primitive_recovery_schemes>>>,
              Actions::UpdateU<>>>,
      Limiters::Actions::SendData<EvolutionMetavars>,
      Limiters::Actions::Limit<EvolutionMetavars>,
      VariableFixing::Actions::FixVariables<grmhd::ValenciaDivClean::Flattener<
          ordered_list_of_primitive_recovery_schemes>>,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives>>;

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      evolution::dg::Actions::ComputeTimeDerivative<volume_dim, system,
                                                    AllStepChoosers>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<Actions::RecordTimeStepperData<>,
                     evolution::Actions::RunEventsAndDenseTriggers<
                         events_and_dense_triggers_subcell_postprocessors>,
                     Actions::UpdateU<>>>,
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
      Actions::RecordTimeStepperData<>,
      evolution::Actions::RunEventsAndDenseTriggers<
          events_and_dense_triggers_subcell_postprocessors>,
      Actions::UpdateU<>,
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
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using dg_registration_list =
      tmpl::list<intrp::Actions::RegisterElementWithInterpolator,
                 observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::flatten<tmpl::list<
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<3>,
      Initialization::Actions::GrTagsForHydro<system>,
      Initialization::Actions::ConservativeSystem<system,
                                                  equation_of_state_tag>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<3, Frame::ElementLogical>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives,

      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::Initialize<
                  volume_dim, system,
                  grmhd::ValenciaDivClean::subcell::DgInitialDataTci>,
              Initialization::Actions::AddSimpleTags<
                  Initialization::subcell::GrTagsForHydro<system, volume_dim>,
                  grmhd::ValenciaDivClean::SetVariablesNeededFixingToFalse>,
              Actions::MutateApply<
                  grmhd::ValenciaDivClean::subcell::SwapGrTags>,
              VariableFixing::Actions::FixVariables<
                  VariableFixing::FixToAtmosphere<volume_dim>>,
              Actions::UpdateConservatives,
              Actions::MutateApply<
                  grmhd::ValenciaDivClean::subcell::SetInitialRdmpData>>,
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
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,

          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::push_back<dg_registration_list,
                              Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>;

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
          use_dg_subcell,
          tmpl::list<
              grmhd::ValenciaDivClean::fd::Tags::Reconstructor,
              ::Tags::VariableFixer<grmhd::ValenciaDivClean::FixConservatives>,
              grmhd::ValenciaDivClean::subcell::Tags::TciOptions>,
          tmpl::list<>>,
      initial_data_tag,
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter>;

  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning.\n\n"};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

struct CenterOfStar : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  struct MaxOfScalar : db::SimpleTag {
    using type = double;
  };

  template <typename TagOfScalar>
  struct MaxOfScalarCompute : db::ComputeTag, MaxOfScalar {
    using base = MaxOfScalar;
    using return_type = double;
    static void function(const gsl::not_null<double*> max_of_scalar,
                         const Scalar<DataVector>& scalar) {
      *max_of_scalar = max(get(scalar));
    };
    using argument_tags = tmpl::list<TagOfScalar>;
  };

  using temporal_id = ::Tags::Time;
  using tags_to_observe =
      tmpl::list<MaxOfScalarCompute<hydro::Tags::RestMassDensity<DataVector>>>;
  using vars_to_interpolate_to_target =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>>;
  using post_interpolation_callback =
      intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe,
                                                   CenterOfStar>;
  using compute_target_points =
      intrp::TargetPoints::SpecifiedPoints<CenterOfStar, 3>;
  using compute_items_on_target = tags_to_observe;

  template <typename Metavariables>
  using interpolating_component =
      typename Metavariables::dg_element_array_component;
};

struct KerrHorizon : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  using temporal_id = ::Tags::Time;
  using tags_to_observe =
      tmpl::list<StrahlkorperTags::EuclideanSurfaceIntegralVectorCompute<
          hydro::Tags::MassFlux<DataVector, 3>, ::Frame::Inertial>>;
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
      intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe,
                                                   KerrHorizon>;

  template <typename Metavariables>
  using interpolating_component =
      typename Metavariables::dg_element_array_component;
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &grmhd::ValenciaDivClean::BoundaryCorrections::register_derived_with_charm,
    &grmhd::ValenciaDivClean::fd::register_derived_with_charm,
    &EquationsOfState::register_derived_with_charm,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
