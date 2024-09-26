// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/Actions/LimitTimeStep.hpp"
#include "ControlSystem/Actions/PrintCurrentMeasurement.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Size/Factory.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Systems/Expansion.hpp"
#include "ControlSystem/Systems/Rotation.hpp"
#include "ControlSystem/Systems/Shape.hpp"
#include "ControlSystem/Systems/Size.hpp"
#include "ControlSystem/Systems/Translation.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"
#include "Domain/FunctionsOfTime/OutputTimeBounds.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteristicSpeeds.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Deadlock/PrintDgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Systems/Cce/Callbacks/DumpBondiSachsOnWorldtube.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
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
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
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
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/ArrayCollection/DgElementCollection.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
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
#include "Parallel/Printf/Printf.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/FilterAction.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ContributeMemoryData.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "ParallelAlgorithms/Amr/Actions/Initialize.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Constraints.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"
#include "ParallelAlgorithms/Amr/Criteria/TruncationError.hpp"
#include "ParallelAlgorithms/Amr/Projectors/CopyFromCreatorOrLeaveAsIs.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Tensors.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Variables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/IgnoreFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeExcisionBoundaryVolumeQuantities.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeExcisionBoundaryVolumeQuantities.tpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.tpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ObserveCenters.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/MonitorMemory.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
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
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylTypeD1.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
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

// We need separate time tags for all horizon finders because of complicated
// Interpolator internal details. So we just make a simple compute tag that
// takes the actual time out of the box since we still want the actual time to
// be the same, just a different tag.
namespace Tags {
template <size_t Index>
struct AhObservationTime {
  static std::string name() { return "AhObservationTime" + get_output(Index); }
  using type = double;
};

template <size_t Index>
struct AhObservationTimeCompute : AhObservationTime<Index>, db::ComputeTag {
  using argument_tags = tmpl::list<::Tags::Time>;
  using base = AhObservationTime<Index>;
  using return_type = double;

  static void function(const gsl::not_null<double*> ah_time,
                       const double time) {
    *ah_time = time;
  }
};
}  // namespace Tags
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

  using initialize_initial_data_dependent_quantities_actions = tmpl::list<
      Actions::MutateApply<gh::gauges::SetPiAndPhiFromConstraints<volume_dim>>,
      Parallel::Actions::TerminatePhase>;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  template <::domain::ObjectLabel Horizon, typename Frame>
  struct Ah : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    static constexpr size_t index = static_cast<size_t>(Horizon);
    using temporal_id = ::Tags::AhObservationTime<index>;
    using vars_to_interpolate_to_target =
        ::ah::vars_to_interpolate_to_target<volume_dim, Frame>;
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using tags_to_observe = ::ah::tags_for_observing<Frame>;
    using surface_tags_to_observe = ::ah::surface_tags_for_observing;
    using compute_items_on_source =
        tmpl::list<::Tags::AhObservationTimeCompute<index>>;
    using compute_items_on_target =
        ::ah::compute_items_on_target<volume_dim, Frame>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<Ah, Frame>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::FindApparentHorizon<Ah, Frame>>;
    using horizon_find_failure_callback =
        intrp::callbacks::IgnoreFailedApparentHorizon;
    using post_horizon_find_callbacks = tmpl::list<
        intrp::callbacks::ObserveSurfaceData<surface_tags_to_observe, Ah,
                                             Frame>,
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, Ah>>;
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
        ah::ComputeExcisionBoundaryVolumeQuantities;
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
                 control_system::Systems::Shape<::domain::ObjectLabel::A, 2,
                                                both_horizons>,
                 control_system::Systems::Shape<::domain::ObjectLabel::B, 2,
                                                both_horizons>,
                 control_system::Systems::Size<::domain::ObjectLabel::A, 2>,
                 control_system::Systems::Size<::domain::ObjectLabel::B, 2>>;

  static constexpr bool use_control_systems =
      tmpl::size<control_systems>::value > 0;

  using interpolator_source_vars = ::ah::source_vars<volume_dim>;
  using source_vars_no_deriv =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
                 gh::Tags::Pi<DataVector, volume_dim>,
                 gh::Tags::Phi<DataVector, volume_dim>>;

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
      gh::gauges::Tags::GaugeAndDerivativeCompute<volume_dim>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            amr::Criterion,
            tmpl::list<
                amr::Criteria::DriveToTarget<volume_dim>,
                amr::Criteria::Constraints<
                    volume_dim,
                    tmpl::list<gh::Tags::ThreeIndexConstraintCompute<
                        volume_dim, Frame::Inertial>>>,
                amr::Criteria::TruncationError<
                    volume_dim, typename system::variables_tag::tags_list>>>,
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
            tmpl::list<::domain::creators::BinaryCompactObject<false>,
                       ::domain::creators::CylindricalBinaryCompactObject>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                intrp::Events::Interpolate<3, AhA, interpolator_source_vars>,
                intrp::Events::Interpolate<3, AhB, interpolator_source_vars>,
                intrp::Events::Interpolate<3, AhC, interpolator_source_vars>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, BondiSachs, source_vars_no_deriv>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, ExcisionBoundaryA, interpolator_source_vars>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, ExcisionBoundaryB, interpolator_source_vars>,
                Events::MonitorMemory<3>, Events::Completion,
                dg::Events::field_observations<volume_dim, observe_fields,
                                               non_tensor_compute_tags>,
                control_system::metafunctions::control_system_events<
                    control_systems>,
                Events::time_events<system>,
                dg::Events::ObserveTimeStepVolume<3>>>>,
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
        // Restrict to monotonic time steppers in LTS to avoid control
        // systems deadlocking.
        tmpl::pair<LtsTimeStepper, TimeSteppers::monotonic_lts_time_steppers>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
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
        tmpl::pair<
            Trigger,
            tmpl::append<Triggers::logical_triggers, Triggers::time_triggers,
                         tmpl::list<Triggers::SeparationLessThan>>>>;
  };

  // A tmpl::list of tags to be added to the GlobalCache by the
  // metavariables
  using const_global_cache_tags =
      tmpl::list<gh::gauges::Tags::GaugeCondition,
                 gh::ConstraintDamping::Tags::DampingFunctionGamma0<
                     volume_dim, Frame::Grid>,
                 gh::ConstraintDamping::Tags::DampingFunctionGamma1<
                     volume_dim, Frame::Grid>,
                 gh::ConstraintDamping::Tags::DampingFunctionGamma2<
                     volume_dim, Frame::Grid>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 intrp::Actions::RegisterElementWithInterpolator>;

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
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                         ::domain::CheckFunctionsOfTimeAreReadyPostprocessor<
                             volume_dim>,
                         evolution::dg::ApplyBoundaryCorrections<
                             local_time_stepping, system, volume_dim, true>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         system, volume_dim, false, use_dg_element_collection>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  system, volume_dim, false, use_dg_element_collection>,
              Actions::RecordTimeStepperData<system>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                  ::domain::CheckFunctionsOfTimeAreReadyPostprocessor<
                      volume_dim>>>,
              control_system::Actions::LimitTimeStep<control_systems>,
              Actions::UpdateU<system>>>,
      Actions::CleanHistory<system, local_time_stepping>,
      dg::Actions::Filter<
          Filters::Exponential<0>,
          tmpl::list<gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
                     gh::Tags::Pi<DataVector, volume_dim>,
                     gh::Tags::Phi<DataVector, volume_dim>>>>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<volume_dim,
                                                use_control_systems>,
          ::amr::Initialization::Initialize<volume_dim>,
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
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      control_system::Actions::InitializeMeasurements<control_systems>,
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
          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<
                  ::domain::Actions::CheckFunctionsOfTimeAreReady<volume_dim>,
                  evolution::Actions::RunEventsAndTriggers,
                  Actions::ChangeSlabSize, step_actions, Actions::AdvanceTime,
                  PhaseControl::Actions::ExecutePhaseChange>>>>>;

  struct BondiSachs : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    static std::string name() { return "BondiSachsInterpolation"; }
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target = source_vars_no_deriv;
    using compute_target_points =
        intrp::TargetPoints::Sphere<BondiSachs, ::Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::DumpBondiSachsOnWorldtube<BondiSachs>>;
    using compute_items_on_target = tmpl::list<>;
    template <typename Metavariables>
    using interpolating_component = gh_dg_element_array;
  };

  using interpolation_target_tags = tmpl::push_back<
      control_system::metafunctions::interpolation_target_tags<control_systems>,
      AhA, AhB, AhC, BondiSachs, ExcisionBoundaryA, ExcisionBoundaryB>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<gh_dg_element_array, dg_registration_list>>;
  };

  using control_components =
      control_system::control_components<EvolutionMetavars, control_systems>;

  static void run_deadlock_analysis_simple_actions(
      Parallel::GlobalCache<EvolutionMetavars>& cache,
      const std::vector<std::string>& deadlocked_components) {
    const auto& functions_of_time =
        Parallel::get<::domain::Tags::FunctionsOfTime>(cache);

    const std::string time_bounds =
        ::domain::FunctionsOfTime::output_time_bounds(functions_of_time);

    Parallel::printf("%s\n", time_bounds);

    if (alg::count(deadlocked_components,
                   pretty_type::name<gh_dg_element_array>()) == 1) {
      tmpl::for_each<control_components>([&cache](auto component_v) {
        using component = tmpl::type_from<decltype(component_v)>;
        Parallel::simple_action<
            control_system::Actions::PrintCurrentMeasurement>(
            Parallel::get_parallel_component<component>(cache));
      });

      if constexpr (Parallel::is_dg_element_collection_v<gh_dg_element_array>) {
        Parallel::threaded_action<Parallel::Actions::SimpleActionOnElement<
            deadlock::PrintElementInfo, true>>(
            Parallel::get_parallel_component<gh_dg_element_array>(cache));
      } else {
        Parallel::simple_action<deadlock::PrintElementInfo>(
            Parallel::get_parallel_component<gh_dg_element_array>(cache));
      }
    }
  }

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = gh_dg_element_array;
    using projectors = tmpl::list<
        Initialization::ProjectTimeStepping<volume_dim>,
        evolution::dg::Initialization::ProjectDomain<volume_dim>,
        Initialization::ProjectTimeStepperHistory<EvolutionMetavars>,
        ::amr::projectors::ProjectVariables<volume_dim,
                                            typename system::variables_tag>,
        evolution::dg::Initialization::ProjectMortars<EvolutionMetavars>,
        evolution::Actions::ProjectRunEventsAndDenseTriggers,
        ::amr::projectors::DefaultInitialize<
            Initialization::Tags::InitialTimeDelta,
            Initialization::Tags::InitialSlabSize<local_time_stepping>,
            ::domain::Tags::InitialExtents<volume_dim>,
            ::domain::Tags::InitialRefinementLevels<volume_dim>,
            evolution::dg::Tags::Quadrature,
            Tags::StepperErrors<typename system::variables_tag>,
            SelfStart::Tags::InitialValue<typename system::variables_tag>,
            SelfStart::Tags::InitialValue<Tags::TimeStep>,
            SelfStart::Tags::InitialValue<Tags::Next<Tags::TimeStep>>,
            evolution::dg::Tags::BoundaryData<volume_dim>>,
        ::amr::projectors::CopyFromCreatorOrLeaveAsIs<tmpl::push_back<
            typename control_system::Actions::InitializeMeasurements<
                control_systems>::simple_tags,
            intrp::Tags::InterpPointInfo<EvolutionMetavars>,
            Tags::ChangeSlabSize::NumberOfExpectedMessages,
            Tags::ChangeSlabSize::NewSlabSize>>>;
  };

  using component_list = tmpl::flatten<tmpl::list<
      ::amr::Component<EvolutionMetavars>,
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      importers::ElementDataReader<EvolutionMetavars>,
      mem_monitor::MemoryMonitor<EvolutionMetavars>,
      intrp::Interpolator<EvolutionMetavars>,
      tmpl::transform<interpolation_target_tags,
                      tmpl::bind<intrp::InterpolationTarget,
                                 tmpl::pin<EvolutionMetavars>, tmpl::_1>>,
      control_system::control_components<EvolutionMetavars, control_systems>,
      gh_dg_element_array>>;

  static constexpr Options::String help{
      "Evolve a binary black hole using the Generalized Harmonic "
      "formulation\n"};
};
