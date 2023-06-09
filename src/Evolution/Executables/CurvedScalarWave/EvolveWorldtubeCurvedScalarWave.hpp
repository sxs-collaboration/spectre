// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Worldtube.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/CurvedScalarWave/CalculateGrVars.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/PsiSquared.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/InitializeConstraintGammas.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/SendToWorldtube.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/FilterAction.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveVolumeIntegrals.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveLineSegment.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/PointInfoTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/PureSphericalHarmonic.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/ByBlock.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

template <typename BackgroundSpacetime, typename InitialData>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = 3;
  using background_spacetime = BackgroundSpacetime;
  static_assert(
      is_analytic_data_v<InitialData> xor is_analytic_solution_v<InitialData>,
      "initial_data must be either an analytic_data or an analytic_solution");

  using system = CurvedScalarWave::System<volume_dim>;
  using temporal_id = Tags::TimeStepId;

  // not implemented yet
  static constexpr bool local_time_stepping = false;

  using analytic_solution_fields = typename system::variables_tag::tags_list;
  using deriv_compute = ::Tags::DerivCompute<
      typename system::variables_tag,
      domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      typename system::gradient_variables>;

  using observe_fields = tmpl::push_back<
      tmpl::flatten<tmpl::list<
          tmpl::append<typename system::variables_tag::tags_list,
                       typename deriv_compute::type::tags_list>,
          CurvedScalarWave::Tags::OneIndexConstraintCompute<volume_dim>,
          CurvedScalarWave::Tags::TwoIndexConstraintCompute<volume_dim>>>,
      domain::Tags::Coordinates<volume_dim, Frame::Grid>,
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
  using non_tensor_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 deriv_compute>;

  template <size_t Number>
  struct PsiAlongAxis
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    static std::string name() {
      return "PsiAlongAxis" + std::to_string(Number);
    }
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<CurvedScalarWave::Tags::Psi,
                   domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::TargetPoints::LineSegment<PsiAlongAxis<Number>, volume_dim,
                                         Frame::Grid>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveLineSegment<vars_to_interpolate_to_target,
                                             PsiAlongAxis<Number>>;
    template <typename metavariables>
    using interpolating_component = typename metavariables::dg_element_array;
  };

  using interpolation_target_tags =
      tmpl::list<PsiAlongAxis<1>, PsiAlongAxis<2>>;
  using interpolator_source_vars =
      tmpl::list<CurvedScalarWave::Tags::Psi,
                 domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            CurvedScalarWave::BoundaryConditions::BoundaryCondition<volume_dim>,
            tmpl::flatten<tmpl::list<
                CurvedScalarWave::BoundaryConditions::
                    standard_boundary_conditions<volume_dim>,
                CurvedScalarWave::BoundaryConditions::Worldtube<volume_dim>>>>,
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<Event, tmpl::flatten<tmpl::list<
                              Events::time_events<system>, Events::Completion,
                              intrp::Events::InterpolateWithoutInterpComponent<
                                  volume_dim, PsiAlongAxis<1>,
                                  EvolutionMetavars, interpolator_source_vars>,
                              intrp::Events::InterpolateWithoutInterpComponent<
                                  volume_dim, PsiAlongAxis<2>,
                                  EvolutionMetavars, interpolator_source_vars>,
                              dg::Events::field_observations<
                                  volume_dim, Tags::Time, observe_fields,
                                  non_tensor_compute_tags>>>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>,
        tmpl::pair<PhaseChange,
                   tmpl::list<PhaseControl::VisitAndReturn<
                                  Parallel::Phase::LoadBalancing>,
                              PhaseControl::CheckpointAndExitAfterWallclock>>,

        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::push_back<StepChoosers::standard_slab_choosers<
                                       system, local_time_stepping>,
                                   StepChoosers::ByBlock<StepChooserUse::Slab,
                                                         volume_dim>>>,
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
  static constexpr bool use_filtering = true;

  struct domain {
    static constexpr bool enable_time_dependent_maps = true;
  };

  using step_actions = tmpl::flatten<tmpl::list<
      CurvedScalarWave::Actions::CalculateGrVars<system>,
      CurvedScalarWave::Worldtube::Actions::SendToWorldtube,
      CurvedScalarWave::Worldtube::Actions::ReceiveWorldtubeData,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping>,
      evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          system, volume_dim, false>,
      Actions::RecordTimeStepperData<system>,
      evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
      Actions::UpdateU<system>,
      tmpl::conditional_t<
          use_filtering,
          dg::Actions::Filter<
              Filters::Exponential<0>,
              tmpl::list<CurvedScalarWave::Tags::Psi,
                         CurvedScalarWave::Tags::Pi,
                         CurvedScalarWave::Tags::Phi<volume_dim>>>,
          tmpl::list<>>>>;

  using const_global_cache_tags = tmpl::list<
      CurvedScalarWave::Tags::BackgroundSpacetime<BackgroundSpacetime>,
      Tags::AnalyticData<InitialData>,
      CurvedScalarWave::Worldtube::Tags::ExcisionSphere<volume_dim>,
      CurvedScalarWave::Worldtube::Tags::ExpansionOrder,
      CurvedScalarWave::Worldtube::Tags::ObserveCoefficientsTrigger>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, local_time_stepping>,
          evolution::dg::Initialization::Domain<volume_dim>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      CurvedScalarWave::Actions::CalculateGrVars<system>,
      Initialization::Actions::AddSimpleTags<
          CurvedScalarWave::Worldtube::Initialization::
              InitializeConstraintDampingGammas<volume_dim>,
          CurvedScalarWave::Initialization::InitializeEvolvedVariables<
              volume_dim>>,
      Initialization::Actions::AddComputeTags<
          StepChoosers::step_chooser_compute_tags<EvolutionMetavars,
                                                  local_time_stepping>>,
      Initialization::Actions::AddComputeTags<tmpl::list<
          CurvedScalarWave::Worldtube::Tags::InertialParticlePositionCompute<
              volume_dim>,
          CurvedScalarWave::Worldtube::Tags::FaceCoordinatesCompute<
              volume_dim, Frame::Grid, true>,
          CurvedScalarWave::Worldtube::Tags::FaceCoordinatesCompute<
              volume_dim, Frame::Inertial, false>,
          CurvedScalarWave::Worldtube::Tags::PunctureFieldCompute<volume_dim>,
          ::domain::Tags::GridToInertialInverseJacobian<volume_dim>>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type =
        std::conditional_t<std::is_same_v<ParallelComponent, dg_element_array>,
                           dg_registration_list, tmpl::list<>>;
  };

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, PsiAlongAxis<1>>,
      intrp::InterpolationTarget<EvolutionMetavars, PsiAlongAxis<2>>,
      CurvedScalarWave::Worldtube::WorldtubeSingleton<EvolutionMetavars>,
      dg_element_array>>;

  static constexpr Options::String help{
      "Evolve a scalar point charge in circular orbit around a Schwarzschild "
      "black hole."};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &CurvedScalarWave::BoundaryCorrections::register_derived_with_charm,
    &register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
