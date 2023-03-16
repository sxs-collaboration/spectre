// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
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
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/ConservativeSystem.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/NumericInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/AllSolutions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiFromGauge.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/SetPiFromGauge.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/FixConservativesAndComputePrims.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/NeighborPackagedData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimsAfterRollback.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/ResizeAndComputePrimitives.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TciOnFdGrid.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/TimeDerivative.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/SetVariablesNeededFixingToFalse.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Evolution/VariableFixing/Actions.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/Actions/ReceiveVolumeData.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/FilterAction.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsOnFailure.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
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
#include "ParallelAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GhGrMhd/Factory.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BlastWave.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BondiHoyleAccretion.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticFieldLoop.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticRotor.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/OrszagTangVortex.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhGrMhd/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/KomissarovShock.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/MassFlux.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/SelfStartActions.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
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

namespace detail {
template <typename InitialData,
          bool IsNumericInitialData =
              evolution::is_numeric_initial_data_v<InitialData>>
struct get_thermodynamic_dim;

template <typename InitialData>
struct get_thermodynamic_dim<InitialData, true> {
  // Use a relativistic 1D EOS for numeric initial data for now. This assumption
  // can be removed once we have an EOS base class that's agnostic to the
  // thermodynamic dimension (alternatively the thermodynamic dimension could be
  // specified as a template parameter to the metavariables, but that's not
  // great because we have to compile separate executables for 1D and 2D EOS).
  static constexpr size_t value = 1;
};

template <typename InitialData>
struct get_thermodynamic_dim<InitialData, false> {
  static constexpr size_t value =
      InitialData::equation_of_state_type::thermodynamic_dim;
};
}  // namespace detail

template <bool UseDgSubcell>
struct GhValenciaDivCleanDefaults {
 public:
  static constexpr size_t volume_dim = 3;
  using domain_frame = Frame::Inertial;
  static constexpr bool use_damped_harmonic_rollon = true;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  using system = grmhd::GhValenciaDivClean::System;
  using analytic_variables_tags =
      typename system::primitive_variables_tag::tags_list;
  using analytic_solution_fields =
      tmpl::append<typename system::primitive_variables_tag::tags_list,
                   typename system::gh_system::variables_tag::tags_list>;
  using ordered_list_of_primitive_recovery_schemes = tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;

  // Do not limit the divergence-cleaning field Phi or the GH fields
  using limiter = Tags::Limiter<
      Limiters::Minmod<3, tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                     grmhd::ValenciaDivClean::Tags::TildeTau,
                                     grmhd::ValenciaDivClean::Tags::TildeS<>,
                                     grmhd::ValenciaDivClean::Tags::TildeB<>>>>;

  using initialize_initial_data_dependent_quantities_actions =
      tmpl::list<Actions::MutateApply<tmpl::conditional_t<
                     UseDgSubcell, grmhd::GhValenciaDivClean::SetPiFromGauge,
                     GeneralizedHarmonic::gauges::SetPiFromGauge<3>>>,
                 Initialization::Actions::AddComputeTags<
                     tmpl::list<gr::Tags::SqrtDetSpatialMetricCompute<
                         volume_dim, domain_frame, DataVector>>>,
                 VariableFixing::Actions::FixVariables<
                     VariableFixing::FixToAtmosphere<volume_dim>>,
                 Actions::UpdateConservatives,
                 Parallel::Actions::TerminatePhase>;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

template <typename EvolutionMetavarsDerived, bool UseDgSubcell>
struct GhValenciaDivCleanTemplateBase;

namespace detail {
template <typename InitialData>
constexpr auto make_default_phase_order() {
  if constexpr (evolution::is_numeric_initial_data_v<InitialData>) {
    return std::array<Parallel::Phase, 8>{
        {Parallel::Phase::Initialization,
         Parallel::Phase::RegisterWithElementDataReader,
         Parallel::Phase::ImportInitialData,
         Parallel::Phase::InitializeInitialDataDependentQuantities,
         Parallel::Phase::InitializeTimeStepperHistory,
         Parallel::Phase::Register, Parallel::Phase::Evolve,
         Parallel::Phase::Exit}};
  } else {
    return std::array<Parallel::Phase, 6>{
        {Parallel::Phase::Initialization,
         Parallel::Phase::InitializeInitialDataDependentQuantities,
         Parallel::Phase::InitializeTimeStepperHistory,
         Parallel::Phase::Register, Parallel::Phase::Evolve,
         Parallel::Phase::Exit}};
  }
}
}  // namespace detail

template <bool UseDgSubcell,
          template <typename, typename...> class EvolutionMetavarsDerived,
          typename InitialData, typename... InterpolationTargetTags>
struct GhValenciaDivCleanTemplateBase<
    EvolutionMetavarsDerived<InitialData, InterpolationTargetTags...>,
    UseDgSubcell> : public virtual GhValenciaDivCleanDefaults<UseDgSubcell> {
  using derived_metavars =
      EvolutionMetavarsDerived<InitialData, InterpolationTargetTags...>;
  using defaults = GhValenciaDivCleanDefaults<UseDgSubcell>;
  static constexpr size_t volume_dim = defaults::volume_dim;
  using domain_frame = typename defaults::domain_frame;
  static constexpr bool use_damped_harmonic_rollon =
      defaults::use_damped_harmonic_rollon;
  using temporal_id = typename defaults::temporal_id;
  static constexpr bool local_time_stepping = defaults::local_time_stepping;
  using system = typename defaults::system;
  using analytic_variables_tags = typename defaults::analytic_variables_tags;
  using analytic_solution_fields = typename defaults::analytic_solution_fields;
  using ordered_list_of_primitive_recovery_schemes =
      typename defaults::ordered_list_of_primitive_recovery_schemes;
  using limiter = typename defaults::limiter;
  using initialize_initial_data_dependent_quantities_actions =
      typename defaults::initialize_initial_data_dependent_quantities_actions;

  static constexpr bool use_dg_subcell = UseDgSubcell;

  using initial_data = InitialData;
  static constexpr bool use_numeric_initial_data =
      evolution::is_numeric_initial_data_v<initial_data>;
  static_assert(
      is_analytic_data_v<initial_data> xor
          is_analytic_solution_v<initial_data> xor use_numeric_initial_data,
      "initial_data must be either an analytic_data, an "
      "analytic_solution, or externally provided numerical initial data");

  static constexpr size_t thermodynamic_dim =
      detail::get_thermodynamic_dim<initial_data>::value;
  // Get the EOS from options for numeric ID, or else from the analytic
  // solution/data.
  using equation_of_state_tag = std::conditional_t<
      use_numeric_initial_data,
      hydro::Tags::EquationOfStateFromOptions<true, thermodynamic_dim>,
      hydro::Tags::EquationOfState<std::unique_ptr<
          EquationsOfState::EquationOfState<true, thermodynamic_dim>>>>;

  using initial_data_list = GeneralizedHarmonic::solutions_including_matter<3>;

  using initial_data_tag =
      tmpl::conditional_t<is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;

  using interpolator_source_vars =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
          typename InterpolationTargetTags::vars_to_interpolate_to_target...>>>;

  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_solution_fields, use_dg_subcell>;
  using error_compute = Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;
  using observe_fields = tmpl::push_back<
      tmpl::append<
          typename system::variables_tag::tags_list,
          typename system::primitive_variables_tag::tags_list,
          tmpl::conditional_t<use_numeric_initial_data, tmpl::list<>,
                              error_tags>,
          tmpl::list<gr::Tags::SpacetimeNormalOneFormCompute<
                         volume_dim, domain_frame, DataVector>,
                     gr::Tags::SpacetimeNormalVectorCompute<
                         volume_dim, domain_frame, DataVector>,
                     gr::Tags::InverseSpacetimeMetricCompute<
                         volume_dim, domain_frame, DataVector>,
                     GeneralizedHarmonic::Tags::GaugeConstraintCompute<
                         volume_dim, domain_frame>,
                     ::Tags::PointwiseL2NormCompute<
                         GeneralizedHarmonic::Tags::GaugeConstraint<
                             volume_dim, domain_frame>>>,
          tmpl::conditional_t<use_dg_subcell,
                              tmpl::list<evolution::dg::subcell::Tags::
                                             TciStatusCompute<volume_dim>>,
                              tmpl::list<>>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::ElementLogical>,
          ::Events::Tags::ObserverCoordinatesCompute<volume_dim,
                                                     Frame::ElementLogical>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<volume_dim,
                                                                   Frame::Grid>,
          ::Events::Tags::ObserverCoordinatesCompute<volume_dim, Frame::Grid>>,
      tmpl::conditional_t<
          use_dg_subcell,
          evolution::dg::subcell::Tags::ObserverCoordinatesCompute<
              volume_dim, Frame::Inertial>,
          ::Events::Tags::ObserverCoordinatesCompute<volume_dim,
                                                     Frame::Inertial>>>;
  using non_tensor_compute_tags = tmpl::append<
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Tags::ObserverMeshCompute<volume_dim>,
              evolution::dg::subcell::Tags::ObserverInverseJacobianCompute<
                  volume_dim, Frame::ElementLogical, Frame::Inertial>,
              evolution::dg::subcell::Tags::ObserverJacobianAndDetInvJacobian<
                  volume_dim, Frame::ElementLogical, Frame::Inertial>>,
          tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                     ::Events::Tags::ObserverInverseJacobianCompute<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::Events::Tags::ObserverJacobianCompute<
                         volume_dim, Frame::ElementLogical, Frame::Inertial>,
                     ::Events::Tags::ObserverDetInvJacobianCompute<
                         Frame::ElementLogical, Frame::Inertial>>>,
      tmpl::conditional_t<use_numeric_initial_data, tmpl::list<>,
                          tmpl::list<analytic_compute, error_compute>>,
      tmpl::list<GeneralizedHarmonic::gauges::Tags::GaugeAndDerivativeCompute<
          volume_dim>>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
   private:
    using boundary_conditions = tmpl::conditional_t<
        use_dg_subcell,
        tmpl::append<
            grmhd::GhValenciaDivClean::BoundaryConditions::
                standard_fd_boundary_conditions,
            tmpl::conditional_t<
                use_numeric_initial_data, tmpl::list<>,
                tmpl::list<grmhd::GhValenciaDivClean::BoundaryConditions::
                               DirichletAnalytic,
                           grmhd::GhValenciaDivClean::BoundaryConditions::
                               DirichletFreeOutflow>>>,
        grmhd::GhValenciaDivClean::BoundaryConditions::
            standard_boundary_conditions>;

   public:
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                dg::Events::field_observations<volume_dim, Tags::Time,
                                               observe_fields,
                                               non_tensor_compute_tags>,
                Events::time_events<system>,
                intrp::Events::Interpolate<3, InterpolationTargetTags,
                                           interpolator_source_vars>...>>>,
        tmpl::pair<
            grmhd::GhValenciaDivClean::BoundaryConditions::BoundaryCondition,
            boundary_conditions>,
        tmpl::pair<GeneralizedHarmonic::gauges::GaugeCondition,
                   GeneralizedHarmonic::gauges::all_gauges>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<PhaseChange, tmpl::list<PhaseControl::VisitAndReturn<
                                    Parallel::Phase::LoadBalancing>>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<
            StepChooser<StepChooserUse::Slab>,
            StepChoosers::standard_slab_choosers<system, local_time_stepping>>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>,
        tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                         Triggers::time_triggers>>>;
  };

  using interpolation_target_tags = tmpl::list<InterpolationTargetTags...>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  using const_global_cache_tags = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              grmhd::GhValenciaDivClean::fd::Tags::Reconstructor,
              grmhd::GhValenciaDivClean::fd::Tags::FilterOptions,
              ::Tags::VariableFixer<grmhd::ValenciaDivClean::FixConservatives>,
              grmhd::ValenciaDivClean::subcell::Tags::TciOptions>,
          tmpl::list<>>,
      GeneralizedHarmonic::gauges::Tags::GaugeCondition,
      tmpl::conditional_t<use_numeric_initial_data, tmpl::list<>,
                          initial_data_tag>,
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter,
      equation_of_state_tag,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma0<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma1<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma2<
          volume_dim, Frame::Grid>>>;

  using dg_registration_list =
      tmpl::list<intrp::Actions::RegisterElementWithInterpolator,
                 observers::Actions::RegisterEventsWithObservers>;

  static constexpr auto default_phase_order =
      detail::make_default_phase_order<initial_data>();

  struct SubcellOptions {
    using evolved_vars_tags = typename system::variables_tag::tags_list;
    using prim_tags = typename system::primitive_variables_tag::tags_list;

    static constexpr bool subcell_enabled = use_dg_subcell;
    static constexpr bool subcell_enabled_at_external_boundary = true;

    // We send `ghost_zone_size` cell-centered grid points for variable
    // reconstruction, of which we need `ghost_zone_size-1` for reconstruction
    // to the internal side of the element face, and `ghost_zone_size` for
    // reconstruction to the external side of the element face.
    template <typename DbTagsList>
    static constexpr size_t ghost_zone_size(
        const db::DataBox<DbTagsList>& box) {
      return db::get<grmhd::GhValenciaDivClean::fd::Tags::Reconstructor>(box)
          .ghost_zone_size();
    }

    using DgComputeSubcellNeighborPackagedData =
        grmhd::GhValenciaDivClean::subcell::NeighborPackagedData;

    using GhostVariables =
        grmhd::GhValenciaDivClean::subcell::PrimitiveGhostVariables;
  };

  using events_and_dense_triggers_subcell_postprocessors =
      tmpl::list<AlwaysReadyPostprocessor<
          grmhd::GhValenciaDivClean::subcell::FixConservativesAndComputePrims<
              ordered_list_of_primitive_recovery_schemes>>>;

  using dg_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::dg::Actions::ApplyLtsBoundaryCorrections<
              system, volume_dim>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  system, volume_dim>,
              Actions::RecordTimeStepperData<>, Actions::UpdateU<>>>,
      Limiters::Actions::SendData<derived_metavars>,
      Limiters::Actions::Limit<derived_metavars>,
      VariableFixing::Actions::FixVariables<
          grmhd::ValenciaDivClean::FixConservatives>,
      Actions::UpdatePrimitives>>;

  using dg_subcell_step_actions = tmpl::flatten<tmpl::list<
      evolution::dg::subcell::Actions::SelectNumericalMethod,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginDg>,
      dg::Actions::Filter<::Filters::Exponential<0>,
                          tmpl::list<gr::Tags::SpacetimeMetric<3>,
                                     GeneralizedHarmonic::Tags::Pi<3>,
                                     GeneralizedHarmonic::Tags::Phi<3>>>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          system, volume_dim>,
      tmpl::conditional_t<
          local_time_stepping, tmpl::list<>,
          tmpl::list<Actions::RecordTimeStepperData<>,
                     evolution::Actions::RunEventsAndDenseTriggers<
                         events_and_dense_triggers_subcell_postprocessors>,
                     Actions::UpdateU<>>>,
      // Note: The primitive variables are computed as part of the TCI.
      evolution::dg::subcell::Actions::TciAndRollback<
          grmhd::GhValenciaDivClean::subcell::TciOnDgGrid<
              tmpl::front<ordered_list_of_primitive_recovery_schemes>>>,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives,
      Actions::Goto<evolution::dg::subcell::Actions::Labels::EndOfSolvers>,

      Actions::Label<evolution::dg::subcell::Actions::Labels::BeginSubcell>,
      evolution::dg::subcell::Actions::SendDataForReconstruction<
          volume_dim,
          grmhd::GhValenciaDivClean::subcell::PrimitiveGhostVariables,
          local_time_stepping>,
      evolution::dg::subcell::Actions::ReceiveDataForReconstruction<volume_dim>,
      Actions::Label<
          evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback>,
      Actions::MutateApply<
          grmhd::GhValenciaDivClean::subcell::PrimsAfterRollback<
              ordered_list_of_primitive_recovery_schemes>>,
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          grmhd::GhValenciaDivClean::subcell::TimeDerivative>,
      Actions::RecordTimeStepperData<>,
      evolution::Actions::RunEventsAndDenseTriggers<
          events_and_dense_triggers_subcell_postprocessors>,
      Actions::UpdateU<>,
      Actions::MutateApply<
          grmhd::GhValenciaDivClean::subcell::FixConservativesAndComputePrims<
              ordered_list_of_primitive_recovery_schemes>>,
      evolution::dg::subcell::Actions::TciAndSwitchToDg<
          grmhd::GhValenciaDivClean::subcell::TciOnFdGrid>,
      Actions::MutateApply<
          grmhd::GhValenciaDivClean::subcell::ResizeAndComputePrims<
              ordered_list_of_primitive_recovery_schemes>>,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives,

      Actions::Label<evolution::dg::subcell::Actions::Labels::EndOfSolvers>>>;

  using step_actions =
      tmpl::conditional_t<use_dg_subcell, dg_subcell_step_actions,
                          dg_step_actions>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<derived_metavars, local_time_stepping>,
          evolution::dg::Initialization::Domain<3>>,
      Initialization::Actions::ConservativeSystem<system>,
      std::conditional_t<
          use_numeric_initial_data, tmpl::list<>,
          evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<volume_dim, Frame::ElementLogical>>>,
      Initialization::Actions::TimeStepperHistory<derived_metavars>,
      VariableFixing::Actions::FixVariables<
          VariableFixing::FixToAtmosphere<volume_dim>>,
      Actions::UpdateConservatives,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,

      tmpl::conditional_t<
          use_dg_subcell,
          tmpl::list<
              evolution::dg::subcell::Actions::Initialize<
                  volume_dim, system,
                  grmhd::GhValenciaDivClean::subcell::DgInitialDataTci>,
              Initialization::Actions::AddSimpleTags<
                  grmhd::ValenciaDivClean::SetVariablesNeededFixingToFalse>,
              VariableFixing::Actions::FixVariables<
                  VariableFixing::FixToAtmosphere<volume_dim>>,
              Actions::UpdateConservatives,
              Actions::MutateApply<
                  grmhd::ValenciaDivClean::subcell::SetInitialRdmpData>>,
          tmpl::list<>>,

      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<
              GhValenciaDivCleanTemplateBase, local_time_stepping>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      Initialization::Actions::Minmod<3>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<derived_metavars>>,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array_component = DgElementArray<
      derived_metavars,
      tmpl::flatten<tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          tmpl::conditional_t<
              use_numeric_initial_data,
              tmpl::list<
                  Parallel::PhaseActions<
                      Parallel::Phase::RegisterWithElementDataReader,
                      tmpl::list<
                          importers::Actions::RegisterWithElementDataReader,
                          Parallel::Actions::TerminatePhase>>,
                  Parallel::PhaseActions<
                      Parallel::Phase::ImportInitialData,
                      tmpl::list<importers::Actions::ReadVolumeData<
                                     evolution::OptionTags::NumericInitialData,
                                     typename system::variables_tag::tags_list>,
                                 importers::Actions::ReceiveVolumeData<
                                     evolution::OptionTags::NumericInitialData,
                                     typename system::variables_tag::tags_list>,
                                 Parallel::Actions::TerminatePhase>>>,
              tmpl::list<>>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<VariableFixing::Actions::FixVariables<
                             VariableFixing::FixToAtmosphere<volume_dim>>,
                         Actions::UpdateConservatives,
                         Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>,
          Parallel::PhaseActions<
              Parallel::Phase::PostFailureCleanup,
              tmpl::list<Actions::RunEventsOnFailure,
                         Parallel::Actions::TerminatePhase>>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, dg_element_array_component>,
        dg_registration_list, tmpl::list<>>;
  };

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<derived_metavars>,
      observers::ObserverWriter<derived_metavars>,
      std::conditional_t<use_numeric_initial_data,
                         importers::ElementDataReader<derived_metavars>,
                         tmpl::list<>>,
      intrp::Interpolator<derived_metavars>,
      intrp::InterpolationTarget<derived_metavars, InterpolationTargetTags>...,
      dg_element_array_component>>;
};
