// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ControlSystem/Actions/InitializeMeasurements.hpp"
#include "ControlSystem/CleanFunctionsOfTime.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Size/Factory.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "ControlSystem/Measurements/SingleHorizon.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Systems/Shape.hpp"
#include "ControlSystem/Systems/Size.hpp"
#include "ControlSystem/Systems/Translation.hpp"
#include "ControlSystem/Trigger.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/Executables/ScalarTensor/ScalarTensorBase.hpp"
#include "Evolution/Systems/Cce/Callbacks/DumpBondiSachsOnWorldtube.hpp"
#include "Evolution/Systems/ScalarTensor/Actions/SetInitialData.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FailedHorizonFind.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/IgnoreFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveCenters.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveFieldsOnHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveTimeSeriesOnHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.tpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Factory.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsOnFailure.hpp"
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
#include "ParallelAlgorithms/Interpolation/ComputeExcisionBoundaryVolumeQuantities.hpp"
#include "ParallelAlgorithms/Interpolation/ComputeExcisionBoundaryVolumeQuantities.tpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"

struct EvolutionMetavars : public ScalarTensorTemplateBase<EvolutionMetavars> {
  using st_base = ScalarTensorTemplateBase<EvolutionMetavars>;
  using typename st_base::initialize_initial_data_dependent_quantities_actions;
  using typename st_base::system;

  static constexpr size_t volume_dim = 3_st;

  static constexpr Options::String help{
      "Evolve the Einstein field equations in GH gauge coupled to a scalar "
      "field \n"
      "on a domain with a single horizon and corresponding excised region"};

  struct ApparentHorizon : tt::ConformsTo<ah::protocols::HorizonMetavars> {
    using time_tag = ah::Tags::ObservationTime<0>;

    using frame = ::Frame::Distorted;

    using horizon_find_callbacks = tmpl::list<
        ah::callbacks::ObserveTimeSeriesOnHorizon<
            ::ah::tags_for_observing<Frame::Distorted>, ApparentHorizon>,
        ah::callbacks::ObserveFieldsOnHorizon<::ah::surface_tags_for_observing,
                                              ApparentHorizon>>;
    using horizon_find_failure_callbacks =
        tmpl::list<ah::callbacks::FailedHorizonFind<ApparentHorizon, false>>;

    using compute_tags_on_element =
        tmpl::list<ah::Tags::ObservationTimeCompute<0>>;

    static constexpr ah::Destination destination = ah::Destination::Observation;

    static std::string name() { return "ApparentHorizon"; }
  };

  struct ExcisionBoundaryA
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
        intrp::TargetPoints::Sphere<ExcisionBoundaryA, ::Frame::Grid>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveSurfaceData<
            tags_to_observe, ExcisionBoundaryA, ::Frame::Grid>>;
    // run_callbacks
    template <typename metavariables>
    using interpolating_component = typename metavariables::st_dg_element_array;
  };

  struct SphericalSurface
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;

    using vars_to_interpolate_to_target =
        detail::ObserverTags::scalar_charge_vars_to_interpolate_to_target;
    using compute_items_on_target =
        detail::ObserverTags::scalar_charge_compute_items_on_target;
    using compute_target_points =
        intrp::TargetPoints::Sphere<SphericalSurface, ::Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            detail::ObserverTags::scalar_charge_surface_obs_tags,
            SphericalSurface>>;
    template <typename metavariables>
    using interpolating_component = typename metavariables::st_dg_element_array;
  };

  using control_systems =
      tmpl::list<control_system::Systems::Shape<
                     ::domain::ObjectLabel::None, 2,
                     control_system::measurements::SingleHorizon<
                         ::domain::ObjectLabel::None>>,
                 control_system::Systems::Translation<
                     2,
                     control_system::measurements::SingleHorizon<
                         ::domain::ObjectLabel::None>,
                     1>,
                 control_system::Systems::Size<::domain::ObjectLabel::None, 2>>;

  static constexpr bool use_control_systems =
      tmpl::size<control_systems>::value > 0;

  struct BondiSachs;

  using interpolation_target_tags = tmpl::push_back<
      control_system::metafunctions::interpolation_target_tags<control_systems>,
      ExcisionBoundaryA, SphericalSurface, BondiSachs>;

  using scalar_charge_interpolator_source_vars =
      detail::ObserverTags::scalar_charge_vars_to_interpolate_to_target;
  using source_vars_no_deriv = tmpl::list<
      gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
      gh::Tags::Pi<DataVector, volume_dim>,
      gh::Tags::Phi<DataVector, volume_dim>, CurvedScalarWave::Tags::Psi,
      CurvedScalarWave::Tags::Pi, CurvedScalarWave::Tags::Phi<volume_dim>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, volume_dim>>;

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
    using interpolating_component = typename Metavariables::st_dg_element_array;
  };
  // The interpolator_source_vars need to be the same in both the
  // Interpolate event and the InterpolateWithoutInterpComponent event.  The
  // Interpolate event interpolates to the horizon, and the
  // InterpolateWithoutInterpComponent event interpolates to the excision
  // boundary. Every Target gets the same interpolator_source_vars, so they need
  // to be made the same. Otherwise a static assert is triggered.
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        // Restrict to monotonic time steppers in LTS to avoid control
        // systems deadlocking.
        tmpl::insert<
            tmpl::erase<typename st_base::factory_creation::factory_classes,
                        LtsTimeStepper>,
            tmpl::pair<LtsTimeStepper,
                       TimeSteppers::monotonic_lts_time_steppers>>,
        tmpl::pair<ah::Criterion, ah::Criteria::standard_criteria>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                ah::Events::FindApparentHorizon<ApparentHorizon>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, BondiSachs, source_vars_no_deriv>,
                control_system::metafunctions::control_system_events<
                    control_systems>,
                control_system::CleanFunctionsOfTime,
                intrp::Events::InterpolateWithoutInterpComponent<
                    volume_dim, ExcisionBoundaryA, ah::source_vars<volume_dim>>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    volume_dim, SphericalSurface,
                    scalar_charge_interpolator_source_vars>>>>,
        tmpl::pair<DenseTrigger,
                   control_system::control_system_triggers<control_systems>>,
        tmpl::pair<control_system::size::State,
                   control_system::size::States::factory_creatable_states>>;
  };

  using typename st_base::const_global_cache_tags;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  using dg_registration_list = typename st_base::dg_registration_list;

  using step_actions = typename st_base::template step_actions<control_systems>;

  using initialization_actions = tmpl::push_back<
      tmpl::pop_back<typename st_base::template initialization_actions<
          use_control_systems>>,
      control_system::Actions::InitializeMeasurements<control_systems>,
      intrp::Actions::ElementInitInterpPoints<volume_dim,
                                              interpolation_target_tags>,
      tmpl::back<typename st_base::template initialization_actions<
          use_control_systems>>>;

  using st_dg_element_array = DgElementArray<
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
          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
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
              Parallel::Phase::Evolve,
              tmpl::flatten<tmpl::list<
                  ::domain::Actions::CheckFunctionsOfTimeAreReady<volume_dim>,
                  std::conditional_t<local_time_stepping,
                                     evolution::Actions::RunEventsAndTriggers<
                                         Triggers::WhenToCheck::AtSteps>,
                                     tmpl::list<>>,
                  evolution::Actions::RunEventsAndTriggers<
                      Triggers::WhenToCheck::AtSlabs>,
                  Actions::ChangeSlabSize, step_actions,
                  Actions::MutateApply<AdvanceTime<>>,
                  PhaseControl::Actions::ExecutePhaseChange>>>,
          Parallel::PhaseActions<
              Parallel::Phase::PostFailureCleanup,
              tmpl::list<Actions::RunEventsOnFailure<::Tags::Time>,
                         Parallel::Actions::TerminatePhase>>>>>;

  // ControlSystem/Measurements/CharSpeed.hpp assumes gh_dg_element_array
  using gh_dg_element_array = st_dg_element_array;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, st_dg_element_array>,
        dg_registration_list, tmpl::list<>>;
  };

  using control_system_horizon_metavars =
      control_system::metafunctions::horizon_metavars<control_systems>;

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      mem_monitor::MemoryMonitor<EvolutionMetavars>,
      importers::ElementDataReader<EvolutionMetavars>, st_dg_element_array,
      ah::Component<EvolutionMetavars, ApparentHorizon>,
      tmpl::transform<
          control_system_horizon_metavars,
          tmpl::bind<ah::Component, tmpl::pin<EvolutionMetavars>, tmpl::_1>>,
      control_system::control_components<EvolutionMetavars, control_systems>,
      tmpl::transform<interpolation_target_tags,
                      tmpl::bind<intrp::InterpolationTarget,
                                 tmpl::pin<EvolutionMetavars>, tmpl::_1>>>>;
};
