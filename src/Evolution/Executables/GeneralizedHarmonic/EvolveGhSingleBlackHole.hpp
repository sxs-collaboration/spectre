// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ApparentHorizons/ComputeExcisionBoundaryVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeExcisionBoundaryVolumeQuantities.tpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.tpp"
#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/HorizonAliases.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/Event.hpp"
#include "ControlSystem/Measurements/SingleHorizon.hpp"
#include "ControlSystem/Systems/Shape.hpp"
#include "ControlSystem/Trigger.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/FunctionsOfTimeAreReady.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
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
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

template <size_t VolumeDim>
struct EvolutionMetavars : public GeneralizedHarmonicTemplateBase<VolumeDim> {
  static constexpr size_t volume_dim = VolumeDim;
  using gh_base = GeneralizedHarmonicTemplateBase<volume_dim>;
  using typename gh_base::initialize_initial_data_dependent_quantities_actions;
  using typename gh_base::system;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation,\n"
      "on a domain with a single horizon and corresponding excised region"};

  struct ApparentHorizon
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using tags_to_observe = ::ah::tags_for_observing<Frame::Inertial>;
    using surface_tags_to_observe = ::ah::surface_tags_for_observing;
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using vars_to_interpolate_to_target =
        ::ah::vars_to_interpolate_to_target<volume_dim, ::Frame::Inertial>;
    using compute_items_on_target =
        ::ah::compute_items_on_target<volume_dim, Frame::Inertial>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<ApparentHorizon,
                                             ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<ApparentHorizon,
                                              ::Frame::Inertial>;
    using horizon_find_failure_callback =
        intrp::callbacks::IgnoreFailedApparentHorizon;
    using post_horizon_find_callbacks = tmpl::list<
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe,
                                                     ApparentHorizon>,
        intrp::callbacks::ObserveSurfaceData<
            surface_tags_to_observe, ApparentHorizon, ::Frame::Inertial>>;
  };

  struct ExcisionBoundary
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using tags_to_observe =
        tmpl::list<gr::Tags::Lapse<DataVector>,
                   gh::ConstraintDamping::Tags::ConstraintGamma1,
                   gh::CharacteristicSpeedsOnStrahlkorper<Frame::Grid>>;
    using compute_vars_to_interpolate =
        ah::ComputeExcisionBoundaryVolumeQuantities;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>,
                   gr::Tags::Shift<DataVector, 3, Frame::Grid>,
                   gr::Tags::SpatialMetric<DataVector, 3, Frame::Grid>,
                   gh::ConstraintDamping::Tags::ConstraintGamma1>;
    using compute_items_on_source = tmpl::list<>;
    using compute_items_on_target = tmpl::append<tmpl::list<
        gr::Tags::DetAndInverseSpatialMetricCompute<DataVector, 3, Frame::Grid>,
        StrahlkorperTags::OneOverOneFormMagnitudeCompute<DataVector, 3,
                                                         Frame::Grid>,
        StrahlkorperTags::UnitNormalOneFormCompute<Frame::Grid>,
        gh::CharacteristicSpeedsOnStrahlkorperCompute<3, Frame::Grid>>>;
    using compute_target_points =
        intrp::TargetPoints::Sphere<ExcisionBoundary, ::Frame::Grid>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveSurfaceData<tags_to_observe, ExcisionBoundary,
                                             ::Frame::Grid>;
    // run_callbacks
    template <typename metavariables>
    using interpolating_component = typename metavariables::gh_dg_element_array;
  };

  using control_systems = tmpl::list<control_system::Systems::Shape<
      ::domain::ObjectLabel::None, 2,
      control_system::measurements::SingleHorizon<
          ::domain::ObjectLabel::None>>>;

  static constexpr bool use_control_systems =
      tmpl::size<control_systems>::value > 0;

  using interpolation_target_tags = tmpl::push_back<
      control_system::metafunctions::interpolation_target_tags<control_systems>,
      ApparentHorizon, ExcisionBoundary>;
  using interpolator_source_vars = ::ah::source_vars<volume_dim>;

  // The interpolator_source_vars need to be the same in both the Interpolate
  // event and the InterpolateWithoutInterpComponent event.  The Interpolate
  // event interpolates to the horizon, and the
  // InterpolateWithoutInterpComponent event interpolates to the excision
  // boundary. Every Target gets the same interpolator_source_vars, so they need
  // to be made the same. Otherwise a static assert is triggered.
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename gh_base::factory_creation::factory_classes,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       intrp::Events::Interpolate<3, ApparentHorizon,
                                                  interpolator_source_vars>,
                       control_system::control_system_events<control_systems>,
                       intrp::Events::InterpolateWithoutInterpComponent<
                           3, ExcisionBoundary, EvolutionMetavars,
                           interpolator_source_vars>>>>,
        tmpl::pair<DenseTrigger,
                   control_system::control_system_triggers<control_systems>>>;
  };

  using typename gh_base::const_global_cache_tags;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          typename ApparentHorizon::post_horizon_find_callbacks,
          typename ExcisionBoundary::post_interpolation_callback>>;

  using dg_registration_list =
      tmpl::push_back<typename gh_base::dg_registration_list,
                      intrp::Actions::RegisterElementWithInterpolator>;

  using typename gh_base::step_actions;

  using initialization_actions = tmpl::push_back<
      tmpl::pop_back<typename gh_base::template initialization_actions<
          EvolutionMetavars, use_control_systems>>,
      control_system::Actions::InitializeMeasurements<control_systems>,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,
      tmpl::back<typename gh_base::template initialization_actions<
          EvolutionMetavars, use_control_systems>>>;

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
          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
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
      mem_monitor::MemoryMonitor<EvolutionMetavars>,
      importers::ElementDataReader<EvolutionMetavars>, gh_dg_element_array,
      intrp::Interpolator<EvolutionMetavars>,
      control_system::control_components<EvolutionMetavars, control_systems>,
      tmpl::transform<interpolation_target_tags,
                      tmpl::bind<intrp::InterpolationTarget,
                                 tmpl::pin<EvolutionMetavars>, tmpl::_1>>>>;
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &gh::BoundaryCorrections::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &gh::ConstraintDamping::register_derived_with_charm,
    &register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
