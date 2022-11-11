// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.tpp"
#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/HorizonAliases.hpp"
#include "ApparentHorizons/ObserveCenters.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Event.hpp"
#include "ControlSystem/Systems/Expansion.hpp"
#include "ControlSystem/Systems/Rotation.hpp"
#include "ControlSystem/Trigger.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/FunctionsOfTimeAreReady.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Protocols/Metavariables.hpp"
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
#include "Evolution/NumericInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/NumericInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DemandOutgoingCharSpeeds.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletMinkowski.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/InitializeDampedHarmonic.hpp"
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
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ContributeMemoryData.hpp"
#include "ParallelAlgorithms/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/MonitorMemory.hpp"
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
#include "ParallelAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/IgnoreFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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

// Note: this executable does not use GeneralizedHarmonicBase.hpp, because
// using it would require a number of changes in GeneralizedHarmonicBase.hpp
// that would apply only when evolving binary black holes. This would
// require adding a number of compile-time switches, an outcome we would prefer
// to avoid.
struct EvolutionMetavars {
  static constexpr size_t volume_dim = 3;
  static constexpr bool use_damped_harmonic_rollon = false;
  using initial_data = evolution::NumericInitialData;
  using system = GeneralizedHarmonic::System<volume_dim>;
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = true;
  // Set override_functions_of_time to true to override the
  // 2nd or 3rd order piecewise polynomial functions of time using
  // `read_spec_piecewise_polynomial()`
  static constexpr bool override_functions_of_time = true;

  using initialize_initial_data_dependent_quantities_actions =
      tmpl::list<GeneralizedHarmonic::gauges::Actions::InitializeDampedHarmonic<
                     volume_dim, use_damped_harmonic_rollon>,
                 Parallel::Actions::TerminatePhase>;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
  struct domain : tt::ConformsTo<::domain::protocols::Metavariables> {
    static constexpr bool enable_time_dependent_maps = true;
  };

  struct AhA : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        ::ah::vars_to_interpolate_to_target<volume_dim, ::Frame::Grid>;
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using tags_to_observe = ::ah::tags_for_observing<Frame::Grid>;
    using surface_tags_to_observe = ::ah::surface_tags_for_observing;
    using compute_items_on_target =
        ::ah::compute_items_on_target<volume_dim, Frame::Grid>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhA, ::Frame::Grid>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhA, ::Frame::Grid>;
    using horizon_find_failure_callback =
        intrp::callbacks::IgnoreFailedApparentHorizon;
    using post_horizon_find_callbacks = tmpl::list<
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, AhA>,
        intrp::callbacks::ObserveSurfaceData<surface_tags_to_observe, AhA,
                                             ::Frame::Grid>,
        ah::callbacks::ObserveCenters<AhA>>;
  };

  struct AhB : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        ::ah::vars_to_interpolate_to_target<volume_dim, ::Frame::Grid>;
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using tags_to_observe = ::ah::tags_for_observing<Frame::Grid>;
    using surface_tags_to_observe = ::ah::surface_tags_for_observing;
    using compute_items_on_target =
        ::ah::compute_items_on_target<volume_dim, Frame::Grid>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhB, ::Frame::Grid>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhB, ::Frame::Grid>;
    using horizon_find_failure_callback =
        intrp::callbacks::IgnoreFailedApparentHorizon;
    using post_horizon_find_callbacks = tmpl::list<
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, AhB>,
        intrp::callbacks::ObserveSurfaceData<surface_tags_to_observe, AhB,
                                             ::Frame::Grid>,
        ah::callbacks::ObserveCenters<AhB>>;
  };

  using control_systems = tmpl::list<control_system::Systems::Rotation<3>,
                                     control_system::Systems::Expansion<2>>;

  static constexpr bool use_control_systems =
      tmpl::size<control_systems>::value > 0;

  using interpolation_target_tags = tmpl::push_back<
      control_system::metafunctions::interpolation_target_tags<control_systems>,
      AhA, AhB>;
  using interpolator_source_vars = ::ah::source_vars<volume_dim>;

  using observe_fields = tmpl::append<
      tmpl::list<
          gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
          gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>,
          gr::Tags::SpacetimeMetric<volume_dim, Frame::Inertial, DataVector>,
          GeneralizedHarmonic::Tags::Pi<volume_dim, Frame::Inertial>,
          GeneralizedHarmonic::Tags::Phi<volume_dim, Frame::Inertial>,
          GeneralizedHarmonic::Tags::GaugeH<volume_dim, Frame::Inertial>,
          GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<volume_dim,
                                                          Frame::Inertial>,
          GeneralizedHarmonic::Tags::GaugeConstraintCompute<volume_dim,
                                                            ::Frame::Inertial>,
          GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<
              volume_dim, ::Frame::Inertial>,
          GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<
              volume_dim, ::Frame::Inertial>,
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
          // observe norms of tensors
          ::Tags::PointwiseL2NormCompute<
              gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>,
          ::Tags::PointwiseL2NormCompute<
              gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>>,
          ::Tags::PointwiseL2NormCompute<gr::Tags::SpacetimeMetric<
              volume_dim, Frame::Inertial, DataVector>>,
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::Pi<volume_dim, Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::Phi<volume_dim, Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::GaugeH<volume_dim, Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<volume_dim,
                                                              Frame::Inertial>>,
          // following tags added to observe constraints
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::GaugeConstraint<volume_dim,
                                                         ::Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::TwoIndexConstraint<volume_dim,
                                                            ::Frame::Inertial>>,
          ::Tags::PointwiseL2NormCompute<
              GeneralizedHarmonic::Tags::ThreeIndexConstraint<
                  volume_dim, ::Frame::Inertial>>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Grid>,
          ::domain::Tags::Coordinates<volume_dim, Frame::Inertial>>,
      // The 4-index constraint is only implemented in 3d
      tmpl::conditional_t<
          volume_dim == 3,
          tmpl::list<
              GeneralizedHarmonic::Tags::
                  FourIndexConstraintCompute<3, ::Frame::Inertial>,
              GeneralizedHarmonic::Tags::FConstraintCompute<3,
                                                            ::Frame::Inertial>,
              ::Tags::PointwiseL2NormCompute<
                  GeneralizedHarmonic::Tags::FConstraint<3, ::Frame::Inertial>>,
              ::Tags::PointwiseL2NormCompute<
                  GeneralizedHarmonic::Tags::FourIndexConstraint<
                      3, ::Frame::Inertial>>,
              GeneralizedHarmonic::Tags::ConstraintEnergyCompute<
                  3, ::Frame::Inertial>>,
          tmpl::list<>>>;
  using non_tensor_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DenseTrigger,
                   tmpl::flatten<tmpl::list<
                       control_system::control_system_triggers<control_systems>,
                       DenseTriggers::standard_dense_triggers>>>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                intrp::Events::Interpolate<3, AhA, interpolator_source_vars>,
                intrp::Events::Interpolate<3, AhB, interpolator_source_vars>,
                Events::MonitorMemory<3, ::Tags::Time>, Events::Completion,
                dg::Events::field_observations<volume_dim, Tags::Time,
                                               observe_fields,
                                               non_tensor_compute_tags>,
                control_system::control_system_events<control_systems>,
                Events::time_events<system>>>>,
        tmpl::pair<GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<
                       volume_dim>,
                   tmpl::list<GeneralizedHarmonic::BoundaryConditions::
                                  ConstraintPreservingBjorhus<volume_dim>,
                              GeneralizedHarmonic::BoundaryConditions::
                                  DirichletMinkowski<volume_dim>,
                              GeneralizedHarmonic::BoundaryConditions::
                                  DemandOutgoingCharSpeeds<volume_dim>>>,
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

  // A tmpl::list of tags to be added to the GlobalCache by the
  // metavariables
  using const_global_cache_tags = tmpl::list<
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma0<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma1<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma2<
          volume_dim, Frame::Grid>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 intrp::Actions::RegisterElementWithInterpolator>;

  static constexpr std::array<Parallel::Phase, 8> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::RegisterWithElementDataReader,
       Parallel::Phase::ImportInitialData,
       Parallel::Phase::InitializeInitialDataDependentQuantities,
       Parallel::Phase::Register, Parallel::Phase::InitializeTimeStepperHistory,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  using step_actions = tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<volume_dim, system,
                                                    AllStepChoosers>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                         ::domain::CheckFunctionsOfTimeAreReadyPostprocessor,
                         evolution::dg::ApplyBoundaryCorrections<
                             EvolutionMetavars, true>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         EvolutionMetavars>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  EvolutionMetavars>,
              Actions::RecordTimeStepperData<>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<
                  ::domain::CheckFunctionsOfTimeAreReadyPostprocessor>>,
              Actions::UpdateU<>>>,
      dg::Actions::Filter<
          Filters::Exponential<0>,
          tmpl::list<
              gr::Tags::SpacetimeMetric<volume_dim, Frame::Inertial,
                                        DataVector>,
              GeneralizedHarmonic::Tags::Pi<volume_dim, Frame::Inertial>,
              GeneralizedHarmonic::Tags::Phi<volume_dim, Frame::Inertial>>>>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<
          volume_dim, override_functions_of_time, use_control_systems>,
      Initialization::Actions::NonconservativeSystem<system>,
      Initialization::Actions::AddComputeTags<::Tags::DerivCompute<
          typename system::variables_tag,
          ::domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                          Frame::Inertial>,
          typename system::gradient_variables>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      Initialization::Actions::AddComputeTags<tmpl::push_back<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars>>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      control_system::Actions::InitializeMeasurements<control_systems>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

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
              tmpl::list<GeneralizedHarmonic::Actions::ReadNumericInitialData<
                             evolution::OptionTags::NumericInitialData>,
                         GeneralizedHarmonic::Actions::SetNumericInitialData<
                             evolution::OptionTags::NumericInitialData>,
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
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<::domain::Actions::CheckFunctionsOfTimeAreReady,
                         Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, gh_dg_element_array>,
        dg_registration_list, tmpl::list<>>;
  };

  using component_list = tmpl::flatten<tmpl::list<
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

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryCorrections::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
