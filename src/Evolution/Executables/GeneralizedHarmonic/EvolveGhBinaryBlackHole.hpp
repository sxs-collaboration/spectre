// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/Actions/LimitTimeStep.hpp"
#include "ControlSystem/CleanFunctionsOfTime.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Size/Factory.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Systems/Expansion.hpp"
#include "ControlSystem/Systems/Rotation.hpp"
#include "ControlSystem/Systems/Shape.hpp"
#include "ControlSystem/Systems/Size.hpp"
#include "ControlSystem/Systems/Skew.hpp"
#include "ControlSystem/Systems/Translation.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteristicSpeeds.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/ChangeFixedLtsRatio.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/FixedLtsRatio.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/NonconformingEqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/ProjectSpectralFilters.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SetupEqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/Deadlock.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Systems/Cce/Callbacks/DumpBondiSachsOnWorldtube.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Callbacks/UpdateCompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionSingleton.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/Events/CheckConstraintThresholds.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/PhaseControl/CheckpointAndExitIfComplete.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletMinkowski.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/SpectralFilter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Triggers/SeparationLessThan.hpp"
#include "Evolution/TypeTraits.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Strahlkorper/IO/InitialShapeFromFile.hpp"
#include "NumericalAlgorithms/Strahlkorper/InitialShape.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/ArrayCollection/DgElementCollection.hpp"
#include "Parallel/ArrayCollection/SimpleActionOnElement.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ContributeMemoryData.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/SpectralFilter.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "ParallelAlgorithms/Amr/Actions/Initialize.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Constraints.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Factory.hpp"
#include "ParallelAlgorithms/Amr/Events/ObserveAmrCriteria.hpp"
#include "ParallelAlgorithms/Amr/Events/ObserveAmrStats.hpp"
#include "ParallelAlgorithms/Amr/Events/RefineMesh.hpp"
#include "ParallelAlgorithms/Amr/Projectors/CopyFromCreatorOrLeaveAsIs.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Tensors.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Variables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FailedHorizonFind.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveCenters.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveFieldsOnHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveTimeSeriesOnHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/SendDependencyToObserverWriter.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Factory.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindCommonHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/KerrSchild.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Events/ChangeFixedLtsRatio.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/MonitorMemory.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsOnFailure.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/ComputeExcisionBoundaryVolumeQuantities.hpp"
#include "ParallelAlgorithms/Interpolation/ComputeExcisionBoundaryVolumeQuantities.tpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylTypeD1.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/ChangeTimeStepperOrder.hpp"
#include "Time/CleanHistory.hpp"
#include "Time/RecordTimeStepperData.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Time/UpdateU.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

// Check if SpEC is linked and therefore we can load SpEC initial data
#ifdef HAS_SPEC_EXPORTER
#include "PointwiseFunctions/AnalyticData/GeneralRelativity/SpecInitialData.hpp"
using SpecInitialData = gr::AnalyticData::SpecInitialData;
#else
using SpecInitialData = NoSuchType;
#endif

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

// Note: this executable does not use GeneralizedHarmonicBase.hpp, because
// using it would require a number of changes in GeneralizedHarmonicBase.hpp
// that would apply only when evolving binary black holes. This would
// require adding a number of compile-time switches, an outcome we would prefer
// to avoid.
struct EvolutionMetavars {
  struct BondiSachs;

  static constexpr size_t volume_dim = 3;
  static constexpr bool use_damped_harmonic_rollon = false;
  using system = gh::System<volume_dim>;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = LtsTimeStepper;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  using initialize_initial_data_dependent_quantities_actions =
      tmpl::list<gh::gauges::SetPiAndPhiFromConstraints<
                     gh::Solutions::all_solutions<volume_dim>, volume_dim>,
                 Parallel::Actions::TerminatePhase>;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  template <::domain::ObjectLabel Horizon, typename Frame>
  struct Ah : tt::ConformsTo<ah::protocols::HorizonMetavars> {
    static constexpr size_t index = 10 + static_cast<size_t>(Horizon);

    using time_tag = ah::Tags::ObservationTime<index>;

    using frame = Frame;

    using horizon_find_callbacks = tmpl::append<
        tmpl::conditional_t<
            Horizon == ::domain::ObjectLabel::C,
            tmpl::list<ah::callbacks::SendDependencyToObserverWriter<Ah, true>,
                       gh::bbh::callbacks::UpdateCompletionCriteria<Ah>>,
            tmpl::list<>>,
        tmpl::list<ah::callbacks::ObserveFieldsOnHorizon<
                       ::ah::surface_tags_for_observing, Ah>,
                   ah::callbacks::ObserveTimeSeriesOnHorizon<
                       ::ah::tags_for_observing<Frame>, Ah>>>;
    using horizon_find_failure_callbacks = tmpl::append<
        // Only ignore errors for AhC
        tmpl::list<ah::callbacks::FailedHorizonFind<
            Ah, Horizon == ::domain::ObjectLabel::C>>,
        tmpl::conditional_t<
            Horizon == ::domain::ObjectLabel::C,
            tmpl::list<
                ah::callbacks::SendDependencyToObserverWriter<Ah, false>>,
            tmpl::list<>>>;

    using compute_tags_on_element =
        tmpl::list<ah::Tags::ObservationTimeCompute<index>>;

    static constexpr ah::Destination destination = ah::Destination::Observation;

    static std::string name() {
      return "ObservationAh" + ::domain::name(Horizon);
    }
  };

  using AhA = Ah<::domain::ObjectLabel::A, ::Frame::Distorted>;
  using AhB = Ah<::domain::ObjectLabel::B, ::Frame::Distorted>;
  using AhC = Ah<::domain::ObjectLabel::C, ::Frame::Inertial>;

  template <::domain::ObjectLabel Excision>
  struct ExcisionBoundary
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using tags_to_observe =
        tmpl::list<gr::Tags::Lapse<DataVector>,
                   gr::Tags::Shift<DataVector, 3, Frame::Grid>>;
    using compute_vars_to_interpolate =
        intrp::ComputeExcisionBoundaryVolumeQuantities;
    using vars_to_interpolate_to_target = tags_to_observe;
    using compute_items_on_source = tmpl::list<>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::TargetPoints::Sphere<ExcisionBoundary<Excision>, ::Frame::Grid>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveSurfaceData<
            tags_to_observe, ExcisionBoundary<Excision>, ::Frame::Grid>>;
    // run_callbacks
    template <typename metavariables>
    using interpolating_component = typename metavariables::gh_dg_element_array;
    static std::string name() {
      return "ObservationExcisionBoundary" + ::domain::name(Excision);
    }
  };

  using ExcisionBoundaryA = ExcisionBoundary<::domain::ObjectLabel::A>;
  using ExcisionBoundaryB = ExcisionBoundary<::domain::ObjectLabel::B>;
  using both_horizons = control_system::measurements::BothHorizons;
  using control_systems =
      tmpl::list<control_system::Systems::Rotation<3, both_horizons>,
                 control_system::Systems::Expansion<2, both_horizons>,
                 control_system::Systems::Translation<2, both_horizons, 2>,
                 control_system::Systems::Skew<2, both_horizons>,
                 control_system::Systems::Shape<::domain::ObjectLabel::A, 2,
                                                both_horizons>,
                 control_system::Systems::Shape<::domain::ObjectLabel::B, 2,
                                                both_horizons>,
                 control_system::Systems::Size<::domain::ObjectLabel::A, 2>,
                 control_system::Systems::Size<::domain::ObjectLabel::B, 2>>;

  static constexpr bool use_control_systems =
      tmpl::size<control_systems>::value > 0;

  using source_vars_no_deriv =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
                 gh::Tags::Pi<DataVector, volume_dim>,
                 gh::Tags::Phi<DataVector, volume_dim>>;

  struct BondiSachs : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    static std::string name() { return "BondiSachsInterpolation"; }
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target = source_vars_no_deriv;
    using compute_target_points =
        intrp::TargetPoints::Sphere<BondiSachs, ::Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::DumpBondiSachsOnWorldtube<BondiSachs>>;
    using compute_items_on_target = tmpl::list<>;
    template <typename metavariables>
    using interpolating_component = typename metavariables::gh_dg_element_array;
  };

  using interpolation_target_tags = tmpl::push_back<
      control_system::metafunctions::interpolation_target_tags<control_systems>,
      BondiSachs, ExcisionBoundaryA, ExcisionBoundaryB>;

  using observe_fields = tmpl::append<
      tmpl::list<
          gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
          gh::Tags::Pi<DataVector, volume_dim>,
          gh::Tags::Phi<DataVector, volume_dim>,
          gh::Tags::GaugeH<DataVector, volume_dim>,
          gh::Tags::SpacetimeDerivGaugeH<DataVector, volume_dim>,
          gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, volume_dim>,
          gr::Tags::SpatialMetric<DataVector, volume_dim>,
          gr::Tags::DetSpatialMetric<DataVector>,
          gr::Tags::InverseSpatialMetric<DataVector, volume_dim>,
          gr::Tags::SqrtDetSpatialMetricCompute<DataVector, volume_dim,
                                                ::Frame::Inertial>,
          gr::Tags::SpacetimeNormalOneFormCompute<DataVector, volume_dim,
                                                  ::Frame::Inertial>,
          gr::Tags::SpacetimeNormalVectorCompute<DataVector, volume_dim,
                                                 ::Frame::Inertial>,
          gr::Tags::InverseSpacetimeMetricCompute<DataVector, volume_dim,
                                                  ::Frame::Inertial>,
          gh::Tags::DerivLapseCompute<volume_dim, Frame::Inertial>,
          gh::Tags::DerivShiftCompute<volume_dim, Frame::Inertial>,
          gh::Tags::DerivSpatialMetricCompute<volume_dim, Frame::Inertial>,
          gr::Tags::DerivInverseSpatialMetricCompute<volume_dim,
                                                     Frame::Inertial>,
          gh::Tags::GaugeConstraintCompute<volume_dim, ::Frame::Inertial>,
          gh::Tags::TwoIndexConstraintCompute<volume_dim, ::Frame::Inertial>,
          gh::Tags::ThreeIndexConstraintCompute<volume_dim, ::Frame::Inertial>,
          gh::Tags::DerivSpatialMetricCompute<volume_dim, ::Frame::Inertial>,
          gr::Tags::SpatialChristoffelFirstKindCompute<DataVector, volume_dim,
                                                       ::Frame::Inertial>,
          gr::Tags::SpatialChristoffelSecondKindCompute<DataVector, volume_dim,
                                                        ::Frame::Inertial>,
          ::Tags::DerivTensorCompute<
              gr::Tags::SpatialChristoffelSecondKind<DataVector, volume_dim>,
              ::domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                              Frame::Inertial>,
              ::domain::Tags::Mesh<volume_dim>>,
          gr::Tags::SpatialRicciCompute<DataVector, volume_dim,
                                        ::Frame::Inertial>,
          gr::Tags::SpatialRicciScalarCompute<DataVector, volume_dim,
                                              ::Frame::Inertial>,
          // observe norms of tensors
          ::Tags::PointwiseL2NormCompute<
              gr::Tags::Shift<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gr::Tags::SpatialMetric<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gr::Tags::SpacetimeMetric<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<gh::Tags::Pi<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<gh::Tags::Phi<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::GaugeH<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::SpacetimeDerivGaugeH<DataVector, volume_dim>>,
          // following tags added to observe constraints
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::GaugeConstraint<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::TwoIndexConstraint<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::ThreeIndexConstraint<DataVector, volume_dim>>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Grid>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Inertial>>,
      // The 4-index constraint is only implemented in 3d
      tmpl::conditional_t<
          volume_dim == 3,
          tmpl::list<
              gh::Tags::FourIndexConstraintCompute<3, ::Frame::Inertial>,
              gh::Tags::FConstraintCompute<3, ::Frame::Inertial>,
              ::Tags::PointwiseL2NormCompute<
                  gh::Tags::FConstraint<DataVector, 3>>,
              ::Tags::PointwiseL2NormCompute<
                  gh::Tags::FourIndexConstraint<DataVector, 3>>,
              gh::Tags::ConstraintEnergyCompute<3, ::Frame::Inertial>,
              gh::Tags::ExtrinsicCurvatureCompute<3, ::Frame::Inertial>,
              ::Tags::DerivTensorCompute<
                  gr::Tags::ExtrinsicCurvature<DataVector, 3>,
                  ::domain::Tags::InverseJacobian<
                      volume_dim, Frame::ElementLogical, Frame::Inertial>,
                  ::domain::Tags::Mesh<volume_dim>>,
              gr::Tags::WeylElectricCompute<DataVector, 3, Frame::Inertial>,
              gr::Tags::WeylElectricScalarCompute<DataVector, 3,
                                                  Frame::Inertial>,
              gr::Tags::WeylTypeD1Compute<DataVector, 3, Frame::Inertial>,
              gr::Tags::WeylTypeD1ScalarCompute<DataVector, 3, Frame::Inertial>,
              gr::Tags::Psi4RealCompute<Frame::Inertial>>,
          tmpl::list<>>>;
  using non_tensor_compute_tags = tmpl::list<
      ::Events::Tags::ObserverMeshCompute<volume_dim>,
      ::Events::Tags::ObserverCoordinatesCompute<volume_dim, Frame::Inertial>,
      ::Events::Tags::ObserverInverseJacobianCompute<
          volume_dim, Frame::ElementLogical, Frame::Inertial>,
      ::Events::Tags::ObserverJacobianCompute<volume_dim, Frame::ElementLogical,
                                              Frame::Inertial>,
      ::Events::Tags::ObserverDetInvJacobianCompute<Frame::ElementLogical,
                                                    Frame::Inertial>,
      ::Events::Tags::ObserverMeshVelocityCompute<volume_dim, Frame::Inertial>,
      gh::gauges::Tags::GaugeAndDerivativeCompute<
          volume_dim, gh::Solutions::all_solutions<volume_dim>>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<ah::Criterion, ah::Criteria::standard_criteria>,
        tmpl::pair<ylm::InitialShape<Frame::Distorted>,
                   tmpl::list<ylm::InitialShapes::Sphere<Frame::Distorted>,
                              ylm::InitialShapes::FromFile<Frame::Distorted>,
                              ah::InitialShapes::KerrSchild<Frame::Distorted>>>,
        tmpl::pair<ylm::InitialShape<Frame::Inertial>,
                   tmpl::list<ylm::InitialShapes::Sphere<Frame::Inertial>,
                              ylm::InitialShapes::FromFile<Frame::Inertial>,
                              ah::InitialShapes::KerrSchild<Frame::Inertial>>>,
        tmpl::pair<
            amr::Criterion,
            tmpl::push_back<
                amr::Criteria::standard_criteria<
                    volume_dim, typename system::variables_tag::tags_list>,
                amr::Criteria::Constraints<
                    volume_dim,
                    tmpl::list<gh::Tags::ThreeIndexConstraintCompute<
                        volume_dim, Frame::Inertial>>>>>,
        tmpl::pair<
            evolution::initial_data::InitialData,
            tmpl::flatten<tmpl::list<
                gh::NumericInitialData,
                tmpl::conditional_t<std::is_same_v<SpecInitialData, NoSuchType>,
                                    tmpl::list<>, SpecInitialData>>>>,
        tmpl::pair<DenseTrigger,
                   tmpl::flatten<tmpl::list<
                       control_system::control_system_triggers<control_systems>,
                       DenseTriggers::standard_dense_triggers>>>,
        tmpl::pair<
            DomainCreator<volume_dim>,
            tmpl::list<::domain::creators::BinaryCompactObject,
                       ::domain::creators::CylindricalBinaryCompactObject>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                ah::Events::FindApparentHorizon<AhA>,
                ah::Events::FindApparentHorizon<AhB>,
                ah::Events::FindCommonHorizon<AhC, observe_fields,
                                              non_tensor_compute_tags>,
                gh::bbh::Events::CheckConstraintThresholds,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, BondiSachs, source_vars_no_deriv>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, ExcisionBoundaryA, ah::source_vars<3>>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, ExcisionBoundaryB, ah::source_vars<3>>,
                Events::MonitorMemory<3>, Events::Completion,
                dg::Events::field_observations<volume_dim, observe_fields,
                                               non_tensor_compute_tags>,
                control_system::metafunctions::control_system_events<
                    control_systems>,
                control_system::CleanFunctionsOfTime,
                Events::time_events<system>,
                dg::Events::ObserveTimeStepVolume<system>,
                amr::Events::RefineMesh,
                amr::Events::ObserveAmrStats<volume_dim>,
                amr::Events::ObserveAmrCriteria<EvolutionMetavars>,
                tmpl::conditional_t<local_time_stepping,
                                    dg::Events::ChangeFixedLtsRatio<volume_dim>,
                                    tmpl::list<>>>>>,
        tmpl::pair<
            evolution::BoundaryCorrection,
            gh::BoundaryCorrections::standard_boundary_corrections<volume_dim>>,
        tmpl::pair<control_system::size::State,
                   control_system::size::States::factory_creatable_states>,
        tmpl::pair<
            gh::BoundaryConditions::BoundaryCondition<volume_dim>,
            tmpl::list<
                gh::BoundaryConditions::ConstraintPreservingBjorhus<volume_dim>,
                gh::BoundaryConditions::DirichletMinkowski<volume_dim>,
                gh::BoundaryConditions::DemandOutgoingCharSpeeds<volume_dim>>>,
        tmpl::pair<
            gh::gauges::GaugeCondition,
            tmpl::list<gh::gauges::DampedHarmonic, gh::gauges::Harmonic>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>,
        // Restrict to monotonic time steppers in LTS to avoid control
        // systems deadlocking.
        tmpl::pair<LtsTimeStepper, TimeSteppers::monotonic_lts_time_steppers>,
        tmpl::pair<PhaseChange,
                   tmpl::push_back<
                       PhaseControl::factory_creatable_classes,
                       gh::bbh::phase_control::CheckpointAndExitIfComplete>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   StepChoosers::standard_step_choosers<system>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::append<StepChoosers::standard_slab_choosers<
                                    system, local_time_stepping>,
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
        tmpl::pair<
            Trigger,
            tmpl::append<Triggers::logical_triggers, Triggers::time_triggers,
                         tmpl::list<Triggers::SeparationLessThan<false>>>>,
        tmpl::pair<Filters::Filter<volume_dim,
                                   typename system::variables_tag::tags_list>,
                   gh::all_filters<volume_dim>>>;
  };

  // A tmpl::list of tags to be added to the GlobalCache by the
  // metavariables
  using const_global_cache_tags =
      tmpl::list<gh::gauges::Tags::GaugeCondition,
                 gh::Tags::DampingFunctionGamma0<volume_dim, Frame::Grid>,
                 gh::Tags::DampingFunctionGamma1<volume_dim, Frame::Grid>,
                 gh::Tags::DampingFunctionGamma2<volume_dim, Frame::Grid>>;

  using mutable_global_cache_tags = tmpl::list<>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using equal_rate_regions =
      tmpl::list<evolution::dg::NonconformingEqualRateRegions<volume_dim>>;

  static constexpr auto default_phase_order =
      std::array{Parallel::Phase::Initialization,
                 Parallel::Phase::RegisterWithElementDataReader,
                 Parallel::Phase::ImportInitialData,
                 Parallel::Phase::InitializeInitialDataDependentQuantities,
                 Parallel::Phase::Register,
                 Parallel::Phase::InitializeTimeStepperHistory,
                 Parallel::Phase::CheckDomain,
                 Parallel::Phase::Evolve,
                 Parallel::Phase::Exit};

  using step_actions = tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          volume_dim, use_dg_element_collection>,
      Actions::MutateApply<RecordTimeStepperData<system>>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                         ::domain::CheckFunctionsOfTimeAreReadyPostprocessor<
                             volume_dim>,
                         evolution::dg::ApplyLtsDenseBoundaryCorrections<
                             EvolutionMetavars>>>,
                     Actions::MutateApply<UpdateU<system, local_time_stepping>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         volume_dim, use_dg_element_collection>,
                     Actions::MutateApply<ChangeTimeStepperOrder<system>>>,
          tmpl::list<
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                  ::domain::CheckFunctionsOfTimeAreReadyPostprocessor<
                      volume_dim>>>,
              control_system::Actions::LimitTimeStep<control_systems>,
              Actions::MutateApply<UpdateU<system, local_time_stepping>>>>,
      Actions::MutateApply<CleanHistory<system>>,
      Actions::MutateApply<evolution::dg::CleanMortarHistory<volume_dim>>,
      dg::Actions::SpectralFilter>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase,
                                       use_control_systems>,
          evolution::dg::Initialization::Domain<EvolutionMetavars,
                                                use_control_systems>,
          ::amr::Initialization::Initialize<volume_dim, EvolutionMetavars>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      Initialization::Actions::AddComputeTags<tmpl::list<::Tags::DerivCompute<
          typename system::variables_tag, ::domain::Tags::Mesh<volume_dim>,
          ::domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                          Frame::Inertial>,
          typename system::gradient_variables>>>,
      gh::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      Initialization::Actions::AddComputeTags<
          tmpl::push_back<StepChoosers::step_chooser_compute_tags<
              EvolutionMetavars, local_time_stepping>>>,
      Initialization::Actions::AddSimpleTags<
          gh::bbh::Actions::InitializeElementCompletionRequested>,
      ::evolution::dg::Initialization::Mortars<volume_dim>,
      intrp::Actions::ElementInitInterpPoints<volume_dim,
                                              interpolation_target_tags>,
      tmpl::conditional_t<
          local_time_stepping,
          evolution::dg::Initialization::Actions::SetupEqualRateRegions<
              EvolutionMetavars, volume_dim, equal_rate_regions>,
          tmpl::list<>>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      control_system::Actions::InitializeMeasurements<control_systems>,
      Initialization::Actions::InitializeItems<
          evolution::dg::Initialization::SpectralFilters<
              volume_dim, typename system::variables_tag::tags_list>>,
      Parallel::Actions::TerminatePhase>;

  using gh_dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::flatten<tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::RegisterWithElementDataReader,
              tmpl::list<importers::Actions::RegisterWithElementDataReader,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::ImportInitialData,
              tmpl::list<gh::Actions::SetInitialData,
                         gh::Actions::ReceiveNumericInitialData,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
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
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::flatten<tmpl::list<
                  ::domain::Actions::CheckFunctionsOfTimeAreReady<volume_dim>,
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
                  PhaseControl::Actions::ExecutePhaseChange>>>,
          Parallel::PhaseActions<
              Parallel::Phase::PostFailureCleanup,
              tmpl::list<Actions::RunEventsOnFailure<::Tags::Time>,
                         Parallel::Actions::TerminatePhase>>>>>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<gh_dg_element_array, dg_registration_list>>;
  };

  using control_system_horizon_metavars =
      control_system::metafunctions::horizon_metavars<control_systems>;
  using control_components =
      control_system::control_components<EvolutionMetavars, control_systems>;

  static void run_deadlock_analysis_simple_actions(
      Parallel::GlobalCache<EvolutionMetavars>& cache,
      const std::vector<std::string>& deadlocked_components) {
    gh::deadlock::run_deadlock_analysis_simple_actions<
        gh_dg_element_array, control_components, interpolation_target_tags,
        control_system_horizon_metavars>(cache, deadlocked_components);
  }

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = gh_dg_element_array;
    using projectors = tmpl::list<
        Initialization::ProjectTimeStepping<volume_dim>,
        evolution::dg::Initialization::ProjectDomain<volume_dim>,
        ::amr::projectors::ProjectVariables<volume_dim,
                                            typename system::variables_tag>,
        evolution::dg::Initialization::ProjectMortars<volume_dim,
                                                      local_time_stepping>,
        Initialization::ProjectTimeStepperHistory<EvolutionMetavars>,
        evolution::Actions::ProjectRunEventsAndDenseTriggers,
        evolution::dg::Initialization::ProjectSpectralFilters<
            volume_dim, typename system::variables_tag::tags_list>,
        ::amr::projectors::DefaultInitialize<
            Initialization::Tags::InitialTimeDelta,
            Initialization::Tags::InitialSlabSize<local_time_stepping>,
            ::domain::Tags::InitialExtents<volume_dim>,
            ::domain::Tags::InitialRefinementLevels<volume_dim>,
            evolution::dg::Tags::Quadrature,
            Tags::StepperErrors<typename system::variables_tag>,
            SelfStart::Tags::InitialValue<typename system::variables_tag>,
            SelfStart::Tags::InitialValue<Tags::TimeStep>>,
        ::amr::projectors::CopyFromCreatorOrLeaveAsIs<tmpl::push_back<
            tmpl::append<
                typename control_system::Actions::InitializeMeasurements<
                    control_systems>::simple_tags,
                tmpl::transform<
                    intrp::InterpolationTarget_detail::
                        get_non_sequential_target_tags<
                            interpolation_target_tags>,
                    tmpl::bind<intrp::Tags::PointInfo, tmpl::_1,
                               tmpl::pin<tmpl::size_t<volume_dim>>>>,
                tmpl::conditional_t<
                    local_time_stepping,
                    tmpl::list<
                        Tags::FixedLtsRatio,
                        Parallel::Tags::Section<
                            gh_dg_element_array,
                            evolution::dg::Tags::EqualRateRegionId>,
                        evolution::dg::Tags::ChangeFixedLtsRatio::
                            NumberOfExpectedMessages,
                        evolution::dg::Tags::ChangeFixedLtsRatio::NewStepSize>,
                    tmpl::list<>>>,
            gh::bbh::Tags::ElementCompletionRequested,
            Tags::ChangeSlabSize::NumberOfExpectedMessages,
            Tags::ChangeSlabSize::NewSlabSize>>>;
    static constexpr bool keep_coarse_grids = false;
    static constexpr bool p_refine_only_in_event = true;
  };

  using component_list = tmpl::flatten<tmpl::list<
      ::amr::Component<EvolutionMetavars>,
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      importers::ElementDataReader<EvolutionMetavars>,
      mem_monitor::MemoryMonitor<EvolutionMetavars>,
      gh::bbh::CompletionSingleton<EvolutionMetavars>,
      ah::Component<EvolutionMetavars, AhA>,
      ah::Component<EvolutionMetavars, AhB>,
      ah::Component<EvolutionMetavars, AhC>,
      tmpl::transform<
          control_system_horizon_metavars,
          tmpl::bind<ah::Component, tmpl::pin<EvolutionMetavars>, tmpl::_1>>,
      tmpl::transform<interpolation_target_tags,
                      tmpl::bind<intrp::InterpolationTarget,
                                 tmpl::pin<EvolutionMetavars>, tmpl::_1>>,
      control_system::control_components<EvolutionMetavars, control_systems>,
      gh_dg_element_array>>;

  static constexpr Options::String help{
      "Evolve a binary black hole using the Generalized Harmonic "
      "formulation\n"};
};
