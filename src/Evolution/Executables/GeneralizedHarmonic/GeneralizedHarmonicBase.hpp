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
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
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
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/TruncationError.hpp"
#include "ParallelAlgorithms/Amr/Projectors/CopyFromCreatorOrLeaveAsIs.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Tensors.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Variables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
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
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylTypeD1.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
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

namespace detail {

template <size_t volume_dim>
struct ObserverTags {
  using system = gh::System<volume_dim>;

  using variables_tag = typename system::variables_tag;
  using analytic_solution_fields = typename variables_tag::tags_list;

  using initial_data_list = gh::Solutions::all_solutions<volume_dim>;

  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_solution_fields, false, initial_data_list>;
  using deriv_compute = ::Tags::DerivCompute<
      variables_tag, domain::Tags::Mesh<volume_dim>,
      domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      typename system::gradient_variables>;
  using error_compute = Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;

  using observe_fields = tmpl::append<
      tmpl::push_back<
          analytic_solution_fields, gh::Tags::GaugeH<DataVector, volume_dim>,
          gh::Tags::SpacetimeDerivGaugeH<DataVector, volume_dim>,
          gr::Tags::SpatialMetric<DataVector, volume_dim>,
          gr::Tags::DetSpatialMetric<DataVector>,
          gr::Tags::InverseSpatialMetric<DataVector, volume_dim>,
          gr::Tags::Shift<DataVector, volume_dim>, gr::Tags::Lapse<DataVector>,
          gr::Tags::SqrtDetSpatialMetricCompute<DataVector, volume_dim,
                                                Frame::Inertial>,
          gr::Tags::SpacetimeNormalOneFormCompute<DataVector, volume_dim,
                                                  Frame::Inertial>,
          gr::Tags::SpacetimeNormalVectorCompute<DataVector, volume_dim,
                                                 Frame::Inertial>,
          gr::Tags::InverseSpacetimeMetricCompute<DataVector, volume_dim,
                                                  Frame::Inertial>,

          gh::Tags::GaugeConstraintCompute<volume_dim, Frame::Inertial>,
          gh::Tags::TwoIndexConstraintCompute<volume_dim, Frame::Inertial>,
          gh::Tags::ThreeIndexConstraintCompute<volume_dim, Frame::Inertial>,
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
          // following tags added to observe constraints
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::GaugeConstraint<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::TwoIndexConstraint<DataVector, volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              gh::Tags::ThreeIndexConstraint<DataVector, volume_dim>>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Grid>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Inertial>>,
      error_tags,
      // The 4-index constraint is only implemented in 3d
      tmpl::conditional_t<
          volume_dim == 3,
          tmpl::list<
              gh::Tags::FourIndexConstraintCompute<3, Frame::Inertial>,
              gh::Tags::FConstraintCompute<3, Frame::Inertial>,
              ::Tags::PointwiseL2NormCompute<
                  gh::Tags::FConstraint<DataVector, 3>>,
              ::Tags::PointwiseL2NormCompute<
                  gh::Tags::FourIndexConstraint<DataVector, 3>>,
              gh::Tags::ConstraintEnergyCompute<3, Frame::Inertial>,
              gh::Tags::ExtrinsicCurvatureCompute<3, Frame::Inertial>,
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
      analytic_compute, error_compute,
      gh::gauges::Tags::GaugeAndDerivativeCompute<volume_dim>>;

  using field_observations =
      dg::Events::field_observations<volume_dim, observe_fields,
                                     non_tensor_compute_tags>;
};

template <size_t volume_dim, bool LocalTimeStepping>
struct FactoryCreation : tt::ConformsTo<Options::protocols::FactoryCreation> {
  using system = gh::System<volume_dim>;

  using factory_classes = tmpl::map<
      tmpl::pair<
          amr::Criterion,
          tmpl::list<
              amr::Criteria::DriveToTarget<volume_dim>,
              amr::Criteria::Constraints<
                  volume_dim, tmpl::list<gh::Tags::ThreeIndexConstraintCompute<
                                  volume_dim, Frame::Inertial>>>,
              amr::Criteria::TruncationError<
                  volume_dim, typename system::variables_tag::tags_list>>>,
      tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
      tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
      tmpl::pair<
          Event,
          tmpl::flatten<tmpl::list<
              Events::Completion, Events::MonitorMemory<volume_dim>,
              typename detail::ObserverTags<volume_dim>::field_observations,
              Events::time_events<system>,
              dg::Events::ObserveTimeStepVolume<volume_dim>>>>,
      tmpl::pair<
          gh::BoundaryConditions::BoundaryCondition<volume_dim>,
          gh::BoundaryConditions::standard_boundary_conditions<volume_dim>>,
      tmpl::pair<gh::gauges::GaugeCondition, gh::gauges::all_gauges>,
      tmpl::pair<
          evolution::initial_data::InitialData,
          tmpl::append<gh::Solutions::all_solutions<volume_dim>,
                       tmpl::conditional_t<volume_dim == 3,
                                           tmpl::list<gh::NumericInitialData>,
                                           tmpl::list<>>>>,
      tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
      tmpl::pair<MathFunction<1, Frame::Inertial>,
                 MathFunctions::all_math_functions<1, Frame::Inertial>>,
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

template <size_t VolumeDim, bool LocalTimeStepping>
struct GeneralizedHarmonicTemplateBase {
  static constexpr size_t volume_dim = VolumeDim;
  using system = gh::System<volume_dim>;
  using TimeStepperBase =
      tmpl::conditional_t<LocalTimeStepping, LtsTimeStepper, TimeStepper>;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  using factory_creation =
      detail::FactoryCreation<volume_dim, local_time_stepping>;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>>>;

  using initialize_initial_data_dependent_quantities_actions = tmpl::list<
      Actions::MutateApply<gh::gauges::SetPiAndPhiFromConstraints<volume_dim>>,
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
                     volume_dim, Frame::Grid>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  // Register needs to be before InitializeTimeStepperHistory so that CCE is
  // properly registered when the self-start happens
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
      dg::Actions::Filter<
          Filters::Exponential<0>,
          tmpl::list<gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
                     gh::Tags::Pi<DataVector, volume_dim>,
                     gh::Tags::Phi<DataVector, volume_dim>>>>;

  template <typename DerivedMetavars, bool UseControlSystems>
  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<DerivedMetavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<volume_dim, UseControlSystems>,
          ::amr::Initialization::Initialize<volume_dim>,
          Initialization::TimeStepperHistory<DerivedMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      Initialization::Actions::AddComputeTags<::Tags::DerivCompute<
          typename system::variables_tag, domain::Tags::Mesh<volume_dim>,
          domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                        Frame::Inertial>,
          typename system::gradient_variables>>,
      gh::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      Initialization::Actions::AddComputeTags<
          tmpl::push_back<StepChoosers::step_chooser_compute_tags<
              GeneralizedHarmonicTemplateBase, local_time_stepping>>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Parallel::Actions::TerminatePhase>;
};
