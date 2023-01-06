// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteristicSpeeds.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/NumericInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/NumericInitialData.hpp"
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
#include "NumericalAlgorithms/LinearOperators/FilterAction.hpp"
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
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
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
#include "ParallelAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
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

template <typename EvolutionMetavarsDerived>
struct GeneralizedHarmonicTemplateBase;

namespace detail {
template <bool UseNumericalInitialData>
constexpr auto make_default_phase_order() {
  if constexpr (UseNumericalInitialData) {
    // Register needs to be before InitializeTimeStepperHistory so that CCE is
    // properly registered when the self-start happens
    return std::array{Parallel::Phase::Initialization,
                      Parallel::Phase::RegisterWithElementDataReader,
                      Parallel::Phase::ImportInitialData,
                      Parallel::Phase::InitializeInitialDataDependentQuantities,
                      Parallel::Phase::Register,
                      Parallel::Phase::InitializeTimeStepperHistory,
                      Parallel::Phase::Evolve,
                      Parallel::Phase::Exit};
  } else {
    return std::array{Parallel::Phase::Initialization,
                      Parallel::Phase::InitializeInitialDataDependentQuantities,
                      Parallel::Phase::Register,
                      Parallel::Phase::InitializeTimeStepperHistory,
                      Parallel::Phase::Evolve,
                      Parallel::Phase::Exit};
  }
}

template <size_t volume_dim>
struct ObserverTags {
  using system = GeneralizedHarmonic::System<volume_dim>;

  using variables_tag = typename system::variables_tag;
  using analytic_solution_fields = typename variables_tag::tags_list;

  using initial_data_list =
      GeneralizedHarmonic::Solutions::all_solutions<volume_dim>;

  using analytic_compute = evolution::Tags::AnalyticSolutionsCompute<
      volume_dim, analytic_solution_fields, false, initial_data_list>;
  using deriv_compute = ::Tags::DerivCompute<
      variables_tag,
      domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      typename system::gradient_variables>;
  using error_compute = Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;

  using observe_fields = tmpl::append<
      tmpl::push_back<
          analytic_solution_fields,
          GeneralizedHarmonic::Tags::GaugeH<volume_dim, Frame::Inertial>,
          GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<volume_dim,
                                                          Frame::Inertial>,
          gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>,
          gr::Tags::DetSpatialMetric<DataVector>,
          gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial,
                                         DataVector>,
          gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
          gr::Tags::Lapse<DataVector>,
          gr::Tags::SqrtDetSpatialMetricCompute<volume_dim, Frame::Inertial,
                                                DataVector>,
          gr::Tags::SpacetimeNormalOneFormCompute<volume_dim, Frame::Inertial,
                                                  DataVector>,
          gr::Tags::SpacetimeNormalVectorCompute<volume_dim, Frame::Inertial,
                                                 DataVector>,
          gr::Tags::InverseSpacetimeMetricCompute<volume_dim, Frame::Inertial,
                                                  DataVector>,

          GeneralizedHarmonic::Tags::GaugeConstraintCompute<volume_dim,
                                                            Frame::Inertial>,
          GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<volume_dim,
                                                               Frame::Inertial>,
          GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<
              volume_dim, Frame::Inertial>,
          GeneralizedHarmonic::Tags::DerivSpatialMetricCompute<
              volume_dim, ::Frame::Inertial>,
          gr::Tags::SpatialChristoffelFirstKindCompute<
              volume_dim, ::Frame::Inertial, DataVector>,
          gr::Tags::SpatialChristoffelSecondKindCompute<
              volume_dim, ::Frame::Inertial, DataVector>,
          ::Tags::DerivTensorCompute<
              gr::Tags::SpatialChristoffelSecondKind<
                  volume_dim, ::Frame::Inertial, DataVector>,
              ::domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                              Frame::Inertial>>,
          gr::Tags::SpatialRicciCompute<volume_dim, ::Frame::Inertial,
                                        DataVector>,
          gr::Tags::SpatialRicciScalarCompute<volume_dim, ::Frame::Inertial,
                                              DataVector>,
          // following tags added to observe constraints
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::GaugeConstraint<volume_dim,
                                                         Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::TwoIndexConstraint<volume_dim,
                                                            Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::ThreeIndexConstraint<volume_dim,
                                                              Frame::Inertial>>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Grid>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Inertial>>,
      error_tags,
      // The 4-index constraint is only implemented in 3d
      tmpl::conditional_t<
          volume_dim == 3,
          tmpl::list<
              GeneralizedHarmonic::Tags::
                  FourIndexConstraintCompute<3, Frame::Inertial>,
              GeneralizedHarmonic::Tags::FConstraintCompute<3, Frame::Inertial>,
              ::Tags::PointwiseL2NormCompute<
                  GeneralizedHarmonic::Tags::FConstraint<3, Frame::Inertial>>,
              ::Tags::PointwiseL2NormCompute<
                  GeneralizedHarmonic::Tags::FourIndexConstraint<
                      3, Frame::Inertial>>,
              GeneralizedHarmonic::Tags::ConstraintEnergyCompute<
                  3, Frame::Inertial>,
              GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<
                  3, Frame::Inertial>,
              ::Tags::DerivTensorCompute<
                  gr::Tags::ExtrinsicCurvature<3, Frame::Inertial>,
                  ::domain::Tags::InverseJacobian<
                      volume_dim, Frame::ElementLogical, Frame::Inertial>>,
              gr::Tags::WeylElectricCompute<3, Frame::Inertial, DataVector>,
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
      GeneralizedHarmonic::gauges::Tags::GaugeAndDerivativeCompute<volume_dim>>;

  using field_observations =
      dg::Events::field_observations<volume_dim, Tags::Time, observe_fields,
                                     non_tensor_compute_tags>;
};

template <size_t volume_dim, bool LocalTimeStepping>
struct FactoryCreation : tt::ConformsTo<Options::protocols::FactoryCreation> {
  using system = GeneralizedHarmonic::System<volume_dim>;

  using factory_classes = tmpl::map<
      tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
      tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
      tmpl::pair<Event,
                 tmpl::flatten<tmpl::list<Events::Completion,
                                          typename detail::ObserverTags<
                                              volume_dim>::field_observations,
                                          Events::time_events<system>>>>,
      tmpl::pair<GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<
                     volume_dim>,
                 GeneralizedHarmonic::BoundaryConditions::
                     standard_boundary_conditions<volume_dim>>,
      tmpl::pair<GeneralizedHarmonic::gauges::GaugeCondition,
                 GeneralizedHarmonic::gauges::all_gauges>,
      tmpl::pair<evolution::initial_data::InitialData,
                 GeneralizedHarmonic::Solutions::all_solutions<volume_dim>>,
      tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
      tmpl::pair<PhaseChange,
                 tmpl::list<PhaseControl::VisitAndReturn<
                                Parallel::Phase::LoadBalancing>,
                            PhaseControl::CheckpointAndExitAfterWallclock>>,
      tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                 StepChoosers::standard_step_choosers<system>>,
      tmpl::pair<
          StepChooser<StepChooserUse::Slab>,
          StepChoosers::standard_slab_choosers<system, LocalTimeStepping>>,
      tmpl::pair<StepController, StepControllers::standard_step_controllers>,
      tmpl::pair<TimeSequence<double>,
                 TimeSequences::all_time_sequences<double>>,
      tmpl::pair<TimeSequence<std::uint64_t>,
                 TimeSequences::all_time_sequences<std::uint64_t>>,
      tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
      tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                       Triggers::time_triggers>>>;
};
}  // namespace detail

template <template <size_t, bool> class EvolutionMetavarsDerived,
          size_t VolumeDim, bool UseNumericalInitialData>
struct GeneralizedHarmonicTemplateBase<
    EvolutionMetavarsDerived<VolumeDim, UseNumericalInitialData>> {

  using derived_metavars =
      EvolutionMetavarsDerived<VolumeDim, UseNumericalInitialData>;
  static constexpr size_t volume_dim = VolumeDim;
  using system = GeneralizedHarmonic::System<volume_dim>;
  static constexpr bool local_time_stepping = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  using factory_creation =
      detail::FactoryCreation<volume_dim, local_time_stepping>;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>>>;

  using initialize_initial_data_dependent_quantities_actions =
      tmpl::list<Actions::MutateApply<
                     GeneralizedHarmonic::gauges::SetPiFromGauge<volume_dim>>,
                 Parallel::Actions::TerminatePhase>;

  // A tmpl::list of tags to be added to the GlobalCache by the
  // metavariables
  using const_global_cache_tags = tmpl::list<
      GeneralizedHarmonic::gauges::Tags::GaugeCondition,
      evolution::initial_data::Tags::InitialData,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma0<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma1<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma2<
          volume_dim, Frame::Grid>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  static constexpr auto default_phase_order =
      detail::make_default_phase_order<UseNumericalInitialData>();

  using step_actions = tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<
                         tmpl::list<evolution::dg::ApplyBoundaryCorrections<
                             local_time_stepping, system, volume_dim, true>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         system, volume_dim>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  system, volume_dim>,
              Actions::RecordTimeStepperData<>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<>,
              dg::Actions::Filter<
                  Filters::Exponential<0>,
                  tmpl::list<gr::Tags::SpacetimeMetric<
                                 volume_dim, Frame::Inertial, DataVector>,
                             GeneralizedHarmonic::Tags::Pi<volume_dim,
                                                           Frame::Inertial>,
                             GeneralizedHarmonic::Tags::Phi<
                                 volume_dim, Frame::Inertial>>>>>>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<derived_metavars, local_time_stepping>,
          evolution::dg::Initialization::Domain<volume_dim>>,
      Initialization::Actions::NonconservativeSystem<system>,
      std::conditional_t<
          UseNumericalInitialData, tmpl::list<>,
          evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<volume_dim, Frame::ElementLogical>>>,
      Initialization::Actions::AddComputeTags<::Tags::DerivCompute<
          typename system::variables_tag,
          domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                        Frame::Inertial>,
          typename system::gradient_variables>>,
      Initialization::Actions::TimeStepperHistory<derived_metavars>,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      Initialization::Actions::AddComputeTags<
          tmpl::push_back<StepChoosers::step_chooser_compute_tags<
              GeneralizedHarmonicTemplateBase, local_time_stepping>>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Parallel::Actions::TerminatePhase>;
};
