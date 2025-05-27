// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/Actions/LimitTimeStep.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Size/Factory.hpp"
#include "ControlSystem/ControlErrors/Size/RegisterDerivedWithCharm.hpp"
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
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteristicSpeeds.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/Deadlock.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Systems/Cce/Callbacks/DumpBondiSachsOnWorldtube.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/PsiSquared.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletMinkowski.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/Actions/SetInitialData.hpp"
#include "Evolution/Systems/ScalarTensor/AllSolutions.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/ScalarTensor/Constraints.hpp"
#include "Evolution/Systems/ScalarTensor/Initialize.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
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
#include "Parallel/ArrayCollection/SimpleActionOnElement.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
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
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/SendDependencyToObserverWriter.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeExcisionBoundaryVolumeQuantities.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeExcisionBoundaryVolumeQuantities.tpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.tpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FindCommonHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ObserveCenters.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/MonitorMemory.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
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
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/AnalyticData/GhScalarTensor/Factory.hpp"
#include "PointwiseFunctions/ConstraintDamping/RegisterDerivedWithCharm.hpp"
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
#include "PointwiseFunctions/ScalarTensor/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/ScalarTensor/ConstraintGammas.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarCharge.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarSource.hpp"
#include "PointwiseFunctions/ScalarTensor/SourceTags.hpp"
#include "PointwiseFunctions/ScalarTensor/StressEnergy.hpp"
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
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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

// Note: this executable follows closely EvolveGhBinaryBlackHole.hpp and does
// not use ScalarTensorBase.hpp.
struct EvolutionMetavars {
  struct BondiSachs;

  static constexpr size_t volume_dim = 3;
  static constexpr bool use_damped_harmonic_rollon = false;
  static constexpr bool backreaction_is_enabled = false;
  using system = ScalarTensor::System;

  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = LtsTimeStepper;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  using initialize_initial_data_dependent_quantities_actions = tmpl::list<
      Actions::MutateApply<gh::gauges::SetPiAndPhiFromConstraints<
          ScalarTensor::InitialData::analytic_solutions_and_data_list,
          volume_dim>>,
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
    using horizon_find_failure_callbacks = tmpl::append<
        tmpl::list<intrp::callbacks::IgnoreFailedApparentHorizon>,
        tmpl::conditional_t<
            Horizon == ::domain::ObjectLabel::C,
            tmpl::list<
                intrp::callbacks::SendDependencyToObserverWriter<false, Ah>>,
            tmpl::list<>>>;
    using post_horizon_find_callbacks = tmpl::append<
        tmpl::conditional_t<
            Horizon == ::domain::ObjectLabel::C,
            tmpl::list<
                intrp::callbacks::SendDependencyToObserverWriter<true, Ah>>,
            tmpl::list<>>,
        tmpl::list<
            intrp::callbacks::ObserveSurfaceData<surface_tags_to_observe, Ah,
                                                 Frame>,
            intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, Ah>>>;
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

  using scalar_charge_interpolator_source_vars = tmpl::list<
      gr::Tags::SpatialMetric<DataVector, volume_dim, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataVector, volume_dim, Frame::Inertial>,
      CurvedScalarWave::Tags::Phi<volume_dim>, CurvedScalarWave::Tags::Psi>;

  template <size_t SphereNumber>
  struct SphericalSurfaceTmp
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;

    using vars_to_interpolate_to_target =
        scalar_charge_interpolator_source_vars;
    using compute_items_on_target = tmpl::list<
        ylm::Tags::ThetaPhiCompute<::Frame::Inertial>,
        ylm::Tags::RadiusCompute<::Frame::Inertial>,
        ylm::Tags::RhatCompute<::Frame::Inertial>,
        ylm::Tags::InvJacobianCompute<::Frame::Inertial>,
        ylm::Tags::JacobianCompute<::Frame::Inertial>,
        ylm::Tags::DxRadiusCompute<::Frame::Inertial>,
        ylm::Tags::NormalOneFormCompute<::Frame::Inertial>,
        ylm::Tags::OneOverOneFormMagnitudeCompute<DataVector, volume_dim,
                                                  ::Frame::Inertial>,
        ylm::Tags::UnitNormalOneFormCompute<::Frame::Inertial>,
        ylm::Tags::UnitNormalVectorCompute<::Frame::Inertial>,
        gr::surfaces::Tags::AreaElementCompute<::Frame::Inertial>,
        ScalarTensor::StrahlkorperScalar::Tags::ScalarChargeIntegrandCompute,
        gr::surfaces::Tags::SurfaceIntegralCompute<
            ScalarTensor::StrahlkorperScalar::Tags::ScalarChargeIntegrand,
            ::Frame::Inertial>,
        gr::surfaces::Tags::SurfaceIntegralCompute<CurvedScalarWave::Tags::Psi,
                                                   ::Frame::Inertial>,
        CurvedScalarWave::Tags::PsiSquaredCompute,
        gr::surfaces::Tags::SurfaceIntegralCompute<
            CurvedScalarWave::Tags::PsiSquared, ::Frame::Inertial>>;
    using compute_target_points =
        intrp::TargetPoints::Sphere<SphericalSurfaceTmp<SphereNumber>,
                                    ::Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<
                gr::surfaces::Tags::SurfaceIntegralCompute<
                    ScalarTensor::StrahlkorperScalar::Tags::
                        ScalarChargeIntegrand,
                    ::Frame::Inertial>,
                gr::surfaces::Tags::SurfaceIntegralCompute<
                    CurvedScalarWave::Tags::Psi, ::Frame::Inertial>,
                gr::surfaces::Tags::SurfaceIntegralCompute<
                    CurvedScalarWave::Tags::PsiSquared, ::Frame::Inertial>>,
            SphericalSurfaceTmp<SphereNumber>>>;
    template <typename metavariables>
    using interpolating_component = typename metavariables::gh_dg_element_array;
    static std::string name() {
      return "SphericalSurface" + std::to_string(SphereNumber);
    }
  };

  using SphericalSurface = SphericalSurfaceTmp<1>;

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

  using interpolator_source_vars = ::ah::source_vars<volume_dim>;

  using source_vars_no_deriv = tmpl::list<
      gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
      gh::Tags::Pi<DataVector, volume_dim>,
      gh::Tags::Phi<DataVector, volume_dim>, CurvedScalarWave::Tags::Psi,
      CurvedScalarWave::Tags::Pi, CurvedScalarWave::Tags::Phi<volume_dim>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, volume_dim>>;

  using observe_fields = tmpl::push_back<
      system::gh_system::variables_tag::tags_list,
      ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::Psi>,
      ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::Pi>,
      ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::Phi<volume_dim>>,
      gh::Tags::GaugeH<DataVector, volume_dim, Frame::Inertial>,
      gh::Tags::SpacetimeDerivGaugeH<DataVector, volume_dim, Frame::Inertial>,
      // 3 plus 1 Tags and derivatives
      gr::Tags::SpatialMetric<DataVector, volume_dim, Frame::Inertial>,
      gr::Tags::DetSpatialMetric<DataVector>,
      gr::Tags::InverseSpatialMetric<DataVector, volume_dim, Frame::Inertial>,
      gr::Tags::Shift<DataVector, volume_dim, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>, gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::SpacetimeNormalOneFormCompute<DataVector, volume_dim,
                                              Frame::Inertial>,
      gr::Tags::SpacetimeNormalVector<DataVector, volume_dim, Frame::Inertial>,
      gr::Tags::InverseSpacetimeMetric<DataVector, volume_dim, Frame::Inertial>,
      ::Tags::deriv<
          gr::Tags::SpatialMetric<DataVector, volume_dim, Frame::Inertial>,
          tmpl::size_t<volume_dim>, Frame::Inertial>,
      gr::Tags::SpatialChristoffelFirstKind<DataVector, volume_dim,
                                            Frame::Inertial>,
      gr::Tags::SpatialChristoffelSecondKind<DataVector, volume_dim,
                                             Frame::Inertial>,
      gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, volume_dim,
                                                  Frame::Inertial>,
      gr::Tags::ExtrinsicCurvature<DataVector, volume_dim, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<
                        DataVector, volume_dim, Frame::Inertial>,
                    tmpl::size_t<volume_dim>, Frame::Inertial>,
      gr::Tags::SpatialRicci<DataVector, volume_dim, Frame::Inertial>,
      gr::Tags::SpatialRicciScalar<DataVector>,
      ::Tags::deriv<
          gr::Tags::ExtrinsicCurvature<DataVector, volume_dim, Frame::Inertial>,
          tmpl::size_t<volume_dim>, Frame::Inertial>,
      gr::Tags::CovariantDerivativeOfExtrinsicCurvature<DataVector, volume_dim,
                                                        Frame::Inertial>,
      // Compute the constraints of GH
      gh::Tags::GaugeConstraintCompute<volume_dim, Frame::Inertial>,
      gh::Tags::TwoIndexConstraintCompute<volume_dim, Frame::Inertial>,
      gh::Tags::ThreeIndexConstraintCompute<volume_dim, Frame::Inertial>,
      gh::Tags::FourIndexConstraintCompute<volume_dim, Frame::Inertial>,
      tmpl::conditional_t<
          backreaction_is_enabled,
          ScalarTensor::Tags::FConstraintCompute<volume_dim, Frame::Inertial>,
          gh::Tags::FConstraintCompute<volume_dim, Frame::Inertial>>,
      // Compute the constraints of CSW
      ScalarTensor::Tags::CswOneIndexConstraintCompute<volume_dim>,
      ScalarTensor::Tags::CswTwoIndexConstraintCompute<volume_dim>,
      // GH constraint norms
      ::Tags::PointwiseL2NormCompute<
          gh::Tags::GaugeConstraint<DataVector, volume_dim, Frame::Inertial>>,
      ::Tags::PointwiseL2NormCompute<gh::Tags::TwoIndexConstraint<
          DataVector, volume_dim, Frame::Inertial>>,
      ::Tags::PointwiseL2NormCompute<gh::Tags::ThreeIndexConstraint<
          DataVector, volume_dim, Frame::Inertial>>,
      ::Tags::PointwiseL2NormCompute<
          gh::Tags::FConstraint<DataVector, volume_dim, Frame::Inertial>>,
      ::Tags::PointwiseL2NormCompute<gh::Tags::FourIndexConstraint<
          DataVector, volume_dim, Frame::Inertial>>,
      gh::Tags::ConstraintEnergyCompute<volume_dim, Frame::Inertial>,
      // CSW constraint norms
      ::Tags::PointwiseL2NormCompute<ScalarTensor::Tags::Csw<
          CurvedScalarWave::Tags::OneIndexConstraint<volume_dim>>>,
      ::Tags::PointwiseL2NormCompute<ScalarTensor::Tags::Csw<
          CurvedScalarWave::Tags::TwoIndexConstraint<volume_dim>>>,
      // Damping parameters
      gh::Tags::ConstraintGamma0, gh::Tags::ConstraintGamma1,
      gh::Tags::ConstraintGamma2,
      ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::ConstraintGamma1>,
      ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::ConstraintGamma2>,
      // Sources
      ScalarTensor::Tags::ScalarSource,
      // Coordinates
      ::domain::Tags::Coordinates<volume_dim, Frame::Grid>,
      ::domain::Tags::Coordinates<volume_dim, Frame::Inertial>,

      gr::Tags::WeylElectric<DataVector, 3, Frame::Inertial>,
      gr::Tags::WeylElectricScalar<DataVector>,
      gr::Tags::WeylMagneticScalar<DataVector>,
      gr::Tags::Psi4RealCompute<Frame::Inertial>>;

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
          volume_dim,
          ScalarTensor::InitialData::analytic_solutions_and_data_list>>;

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
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::flatten<tmpl::list<ScalarTensor::NumericInitialData>>>,
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
                ah::Events::FindCommonHorizon<
                    volume_dim, AhC, interpolator_source_vars, observe_fields,
                    non_tensor_compute_tags>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, BondiSachs, source_vars_no_deriv>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, ExcisionBoundaryA, interpolator_source_vars>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, ExcisionBoundaryB, interpolator_source_vars>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, SphericalSurface,
                    scalar_charge_interpolator_source_vars>,
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
            ScalarTensor::BoundaryConditions::BoundaryCondition,
            ScalarTensor::BoundaryConditions::standard_boundary_conditions>,
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
                         tmpl::list<Triggers::SeparationLessThan<false>>>>>;
  };

  // A tmpl::list of tags to be added to the GlobalCache by the
  // metavariables
  using const_global_cache_tags = tmpl::list<
      gh::gauges::Tags::GaugeCondition,
      gh::Tags::DampingFunctionGamma0<volume_dim, Frame::Grid>,
      gh::Tags::DampingFunctionGamma1<volume_dim, Frame::Grid>,
      gh::Tags::DampingFunctionGamma2<volume_dim, Frame::Grid>,
      ScalarTensor::Tags::DampingFunctionGamma1<volume_dim, Frame::Grid>,
      ScalarTensor::Tags::DampingFunctionGamma2<volume_dim, Frame::Grid>,
      // Source parameters
      ScalarTensor::Tags::ScalarMass, ScalarTensor::Tags::CouplingParameters,
      ScalarTensor::Tags::RampUpParameters>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 intrp::Actions::RegisterElementWithInterpolator>;

  static constexpr std::array<Parallel::Phase, 9> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::RegisterWithElementDataReader,
       Parallel::Phase::ImportInitialData,
       Parallel::Phase::InitializeInitialDataDependentQuantities,
       Parallel::Phase::Register, Parallel::Phase::InitializeTimeStepperHistory,
       Parallel::Phase::CheckDomain, Parallel::Phase::Evolve,
       Parallel::Phase::Exit}};

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
                     gh::Tags::Phi<DataVector, volume_dim>>>,
      dg::Actions::Filter<Filters::Exponential<1>,
                          system::scalar_system::variables_tag::tags_list>>;

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
      ScalarTensor::Actions::InitializeGhAnd3Plus1Variables,
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
              tmpl::list<ScalarTensor::Actions::SetInitialData,
                         ScalarTensor::Actions::ReceiveNumericInitialData,
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
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<
                  ::domain::Actions::CheckFunctionsOfTimeAreReady<volume_dim>,
                  evolution::Actions::RunEventsAndTriggers<local_time_stepping>,
                  Actions::ChangeSlabSize, step_actions, Actions::AdvanceTime,
                  PhaseControl::Actions::ExecutePhaseChange>>,
          Parallel::PhaseActions<
              Parallel::Phase::PostFailureCleanup,
              tmpl::list<Actions::RunEventsOnFailure<::Tags::Time>,
                         Parallel::Actions::TerminatePhase>>>>>;

  struct BondiSachs : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    static std::string name() { return "BondiSachsInterpolation"; }
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target = source_vars_no_deriv;
    using compute_target_points =
        intrp::TargetPoints::Sphere<BondiSachs, ::Frame::Inertial>;
    using post_interpolation_callbacks = tmpl::list<
        intrp::callbacks::DumpBondiSachsOnWorldtube<BondiSachs, true>>;
    using compute_items_on_target = tmpl::list<>;
    template <typename Metavariables>
    using interpolating_component = gh_dg_element_array;
  };

  using interpolation_target_tags = tmpl::push_back<
      control_system::metafunctions::interpolation_target_tags<control_systems>,
      AhA, AhB, AhC, BondiSachs, ExcisionBoundaryA, ExcisionBoundaryB,
      SphericalSurface>;

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
    gh::deadlock::run_deadlock_analysis_simple_actions<
        gh_dg_element_array, control_components, interpolation_target_tags>(
        cache, deadlocked_components);
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
      "Evolve a binary black hole with a test-field scalar field.\n"};
};
