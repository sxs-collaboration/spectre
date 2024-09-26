// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "ControlSystem/Actions/LimitTimeStep.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteristicSpeeds.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/PsiSquared.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/ProductOfConditions.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/ScalarTensor/Constraints.hpp"
#include "Evolution/Systems/ScalarTensor/Initialize.hpp"
#include "Evolution/Systems/ScalarTensor/Sources/ScalarSource.hpp"
#include "Evolution/Systems/ScalarTensor/StressEnergy.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
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
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/FilterAction.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ContributeMemoryData.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/RandomizeVariables.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/MonitorMemory.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/GhScalarTensor/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarCharge.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
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

template <typename EvolutionMetavarsDerived>
struct ScalarTensorTemplateBase;

namespace detail {
constexpr auto make_default_phase_order() {
  return std::array{Parallel::Phase::Initialization,
                    Parallel::Phase::InitializeInitialDataDependentQuantities,
                    Parallel::Phase::Register,
                    Parallel::Phase::InitializeTimeStepperHistory,
                    Parallel::Phase::Evolve,
                    Parallel::Phase::Exit};
}

struct ObserverTags {
  static constexpr size_t volume_dim = 3_st;

  using system = ScalarTensor::System;

  using variables_tag = typename system::variables_tag;

  using initial_data_list = gh::ScalarTensor::AnalyticData::all_analytic_data;

  using deriv_compute = ::Tags::DerivCompute<
      variables_tag, domain::Tags::Mesh<volume_dim>,
      domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      typename system::gradient_variables>;

  using observe_fields = tmpl::append<
      tmpl::push_back<
          system::gh_system::variables_tag::tags_list,
          ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::Psi>,
          ScalarTensor::Tags::CswCompute<CurvedScalarWave::Tags::Pi>,
          ScalarTensor::Tags::CswCompute<
              CurvedScalarWave::Tags::Phi<volume_dim>>,
          gh::Tags::GaugeH<DataVector, volume_dim, Frame::Inertial>,
          gh::Tags::SpacetimeDerivGaugeH<DataVector, volume_dim,
                                         Frame::Inertial>,
          // 3 plus 1 Tags and derivatives
          gr::Tags::SpatialMetric<DataVector, volume_dim, Frame::Inertial>,
          gr::Tags::DetSpatialMetric<DataVector>,
          gr::Tags::InverseSpatialMetric<DataVector, volume_dim,
                                         Frame::Inertial>,
          gr::Tags::Shift<DataVector, volume_dim, Frame::Inertial>,
          gr::Tags::Lapse<DataVector>,
          gr::Tags::SqrtDetSpatialMetricCompute<DataVector, volume_dim,
                                                Frame::Inertial>,
          gr::Tags::SpacetimeNormalOneFormCompute<DataVector, volume_dim,
                                                  Frame::Inertial>,
          gr::Tags::SpacetimeNormalVector<DataVector, volume_dim,
                                          Frame::Inertial>,
          gr::Tags::InverseSpacetimeMetric<DataVector, volume_dim,
                                           Frame::Inertial>,
          ::Tags::deriv<
              gr::Tags::SpatialMetric<DataVector, volume_dim, Frame::Inertial>,
              tmpl::size_t<volume_dim>, Frame::Inertial>,
          gr::Tags::SpatialChristoffelFirstKind<DataVector, volume_dim,
                                                Frame::Inertial>,
          gr::Tags::SpatialChristoffelSecondKind<DataVector, volume_dim,
                                                 Frame::Inertial>,
          // 3 plus 1 variables used by CSW
          gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, volume_dim,
                                                      Frame::Inertial>,
          gr::Tags::ExtrinsicCurvature<DataVector, volume_dim, Frame::Inertial>,
          gr::Tags::TraceExtrinsicCurvature<DataVector>,
          // Compute the constraints of GH
          gh::Tags::GaugeConstraintCompute<volume_dim, Frame::Inertial>,
          gh::Tags::TwoIndexConstraintCompute<volume_dim, Frame::Inertial>,
          gh::Tags::ThreeIndexConstraintCompute<volume_dim, Frame::Inertial>,
          ::Tags::DerivTensorCompute<
              gr::Tags::SpatialChristoffelSecondKind<DataVector, volume_dim>,
              ::domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                              Frame::Inertial>,
              ::domain::Tags::Mesh<volume_dim>>,
          gr::Tags::SpatialRicciCompute<DataVector, volume_dim,
                                        ::Frame::Inertial>,
          gr::Tags::SpatialRicciScalarCompute<DataVector, volume_dim,
                                              ::Frame::Inertial>,
          // Compute the constraints of CSW
          ScalarTensor::Tags::CswOneIndexConstraintCompute<volume_dim>,
          ScalarTensor::Tags::CswTwoIndexConstraintCompute<volume_dim>,
          // GH constraint norms
          ::Tags::PointwiseL2NormCompute<gh::Tags::GaugeConstraint<
              DataVector, volume_dim, Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<gh::Tags::TwoIndexConstraint<
              DataVector, volume_dim, Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<gh::Tags::ThreeIndexConstraint<
              DataVector, volume_dim, Frame::Inertial>>,
          // CSW constraint norms
          ::Tags::PointwiseL2NormCompute<ScalarTensor::Tags::Csw<
              CurvedScalarWave::Tags::OneIndexConstraint<volume_dim>>>,
          ::Tags::PointwiseL2NormCompute<ScalarTensor::Tags::Csw<
              CurvedScalarWave::Tags::TwoIndexConstraint<volume_dim>>>,
          // Damping parameters
          gh::ConstraintDamping::Tags::ConstraintGamma0,
          gh::ConstraintDamping::Tags::ConstraintGamma1,
          gh::ConstraintDamping::Tags::ConstraintGamma2,
          // Sources
          ScalarTensor::Tags::TraceReversedStressEnergyCompute,
          ScalarTensor::Tags::ScalarSource,

          ::domain::Tags::Coordinates<volume_dim, Frame::Grid>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Inertial>>,
      // The 4-index constraint is only implemented in 3d
      tmpl::list<
          gh::Tags::FourIndexConstraintCompute<volume_dim, Frame::Inertial>,
          ScalarTensor::Tags::FConstraintCompute<volume_dim, Frame::Inertial>,
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::FConstraint<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::FourIndexConstraint<DataVector, volume_dim>>,
          gh::Tags::ConstraintEnergyCompute<volume_dim, Frame::Inertial>,
          ::Tags::DerivTensorCompute<
              gr::Tags::ExtrinsicCurvature<DataVector, volume_dim>,
              ::domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                              Frame::Inertial>,
              ::domain::Tags::Mesh<volume_dim>>,
          gr::Tags::WeylElectricCompute<DataVector, volume_dim,
                                        Frame::Inertial>,
          gr::Tags::Psi4RealCompute<Frame::Inertial>>>;
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

  using field_observations =
      dg::Events::field_observations<volume_dim, observe_fields,
                                     non_tensor_compute_tags>;

  // We collect here all the tags needed for interpolation in all surfaces
  using scalar_charge_vars_to_interpolate_to_target = tmpl::list<
      gr::Tags::SpatialMetric<DataVector, volume_dim, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataVector, volume_dim, Frame::Inertial>,
      CurvedScalarWave::Tags::Phi<volume_dim>, CurvedScalarWave::Tags::Psi>;

  using scalar_charge_compute_items_on_target = tmpl::list<
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

  using scalar_charge_surface_obs_tags = tmpl::list<
      gr::surfaces::Tags::SurfaceIntegralCompute<
          ScalarTensor::StrahlkorperScalar::Tags::ScalarChargeIntegrand,
          ::Frame::Inertial>,
      gr::surfaces::Tags::SurfaceIntegralCompute<CurvedScalarWave::Tags::Psi,
                                                 ::Frame::Inertial>,
      gr::surfaces::Tags::SurfaceIntegralCompute<
          CurvedScalarWave::Tags::PsiSquared, ::Frame::Inertial>>;
};

template <bool LocalTimeStepping>
struct FactoryCreation : tt::ConformsTo<Options::protocols::FactoryCreation> {
  static constexpr size_t volume_dim = 3_st;

  using system = ScalarTensor::System;

  using initial_data_list = gh::ScalarTensor::AnalyticData::all_analytic_data;
  using factory_classes = tmpl::map<
      tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
      tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
      tmpl::pair<Event,
                 tmpl::flatten<tmpl::list<
                     Events::Completion, Events::MonitorMemory<volume_dim>,
                     typename detail::ObserverTags::field_observations,
                     Events::time_events<system>,
                     dg::Events::ObserveTimeStepVolume<volume_dim>>>>,
      tmpl::pair<
          ScalarTensor::BoundaryConditions::BoundaryCondition,
          ScalarTensor::BoundaryConditions::standard_boundary_conditions>,
      tmpl::pair<gh::gauges::GaugeCondition, gh::gauges::all_gauges>,
      tmpl::pair<evolution::initial_data::InitialData, initial_data_list>,
      tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
      tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
      tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                 StepChoosers::standard_step_choosers<system>>,
      tmpl::pair<
          StepChooser<StepChooserUse::Slab>,
          StepChoosers::standard_slab_choosers<system, LocalTimeStepping>>,
      tmpl::pair<TimeSequence<double>,
                 TimeSequences::all_time_sequences<double>>,
      tmpl::pair<TimeSequence<std::uint64_t>,
                 TimeSequences::all_time_sequences<std::uint64_t>>,
      tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
      tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                       Triggers::time_triggers>>>;
};
}  // namespace detail

template <class EvolutionMetavarsDerived>
struct ScalarTensorTemplateBase {
  using derived_metavars = EvolutionMetavarsDerived;

  static constexpr size_t volume_dim = 3_st;
  using system = ScalarTensor::System;
  using TimeStepperBase = LtsTimeStepper;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  using factory_creation = detail::FactoryCreation<local_time_stepping>;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>>>;

  using initialize_initial_data_dependent_quantities_actions = tmpl::list<
      Initialization::Actions::AddComputeTags<
          ScalarTensor::Initialization::scalar_tensor_3plus1_compute_tags<
              volume_dim>>,
      Actions::MutateApply<gh::gauges::SetPiAndPhiFromConstraints<volume_dim>>,
      Initialization::Actions::AddSimpleTags<
          CurvedScalarWave::Initialization::InitializeConstraintDampingGammas<
              volume_dim>>,
      Parallel::Actions::TerminatePhase>;

  // A tmpl::list of tags to be added to the GlobalCache by the
  // metavariables
  using const_global_cache_tags =
      tmpl::list<gh::gauges::Tags::GaugeCondition,
                 evolution::initial_data::Tags::InitialData,
                 gh::ConstraintDamping::Tags::DampingFunctionGamma0<
                     volume_dim, Frame::Grid>,
                 gh::ConstraintDamping::Tags::DampingFunctionGamma1<
                     volume_dim, Frame::Grid>,
                 gh::ConstraintDamping::Tags::DampingFunctionGamma2<
                     volume_dim, Frame::Grid>,
                 // Source parameters
                 ScalarTensor::Tags::ScalarMass>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  static constexpr auto default_phase_order =
      detail::make_default_phase_order();

  template <typename ControlSystems>
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
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              control_system::Actions::LimitTimeStep<ControlSystems>,
              Actions::UpdateU<system>>>,
      Actions::CleanHistory<system, local_time_stepping>,
      // We allow for separate filtering of the system variables
      dg::Actions::Filter<Filters::Exponential<0>,
                          system::gh_system::variables_tag::tags_list>,
      dg::Actions::Filter<Filters::Exponential<1>,
                          system::scalar_system::variables_tag::tags_list>>;

  //   // For labeling the yaml option for RandomizeVariables
  //   struct RandomizeInitialGuess {};

  template <bool UseControlSystems>
  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<derived_metavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<volume_dim, UseControlSystems>,
          Initialization::TimeStepperHistory<derived_metavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<volume_dim, Frame::ElementLogical>>,
      // Random noise system::variables_tag
      //   Actions::RandomizeVariables<typename system::variables_tag,
      //                               RandomizeInitialGuess>,
      Initialization::Actions::AddComputeTags<::Tags::DerivCompute<
          typename system::variables_tag, domain::Tags::Mesh<volume_dim>,
          domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                        Frame::Inertial>,
          typename system::gradient_variables>>,
      Initialization::Actions::AddComputeTags<
          tmpl::push_back<StepChoosers::step_chooser_compute_tags<
              ScalarTensorTemplateBase, local_time_stepping>>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Parallel::Actions::TerminatePhase>;
};
