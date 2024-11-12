// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/CalculateGrVars.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/PsiSquared.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/FilterAction.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/Factory.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
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
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/StepChoosers/ByBlock.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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

template <size_t Dim, typename BackgroundSpacetime, typename InitialData>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;
  using background_spacetime = BackgroundSpacetime;
  static_assert(
      is_analytic_data_v<InitialData> xor is_analytic_solution_v<InitialData>,
      "initial_data must be either an analytic_data or an analytic_solution");

  using system = CurvedScalarWave::System<Dim>;
  using temporal_id = Tags::TimeStepId;
  using TimeStepperBase = LtsTimeStepper;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  using analytic_solution_fields = typename system::variables_tag::tags_list;
  using deriv_compute = ::Tags::DerivCompute<
      typename system::variables_tag, domain::Tags::Mesh<volume_dim>,
      domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      typename system::gradient_variables>;

  using observe_fields = tmpl::push_back<
      tmpl::flatten<tmpl::list<
          tmpl::append<typename system::variables_tag::tags_list,
                       typename deriv_compute::type::tags_list>,
          ::Events::Tags::ObserverDetInvJacobianCompute<Frame::ElementLogical,
                                                        Frame::Inertial>,
          CurvedScalarWave::Tags::OneIndexConstraintCompute<volume_dim>,
          CurvedScalarWave::Tags::TwoIndexConstraintCompute<volume_dim>,
          ::Tags::PointwiseL2NormCompute<
              CurvedScalarWave::Tags::OneIndexConstraint<volume_dim>>,
          ::Tags::PointwiseL2NormCompute<
              CurvedScalarWave::Tags::TwoIndexConstraint<volume_dim>>>>,
      domain::Tags::Coordinates<volume_dim, Frame::Grid>,
      domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
  using non_tensor_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 deriv_compute>;

  static constexpr bool interpolate = volume_dim == 3;
  struct SphericalSurface
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpatialMetric<DataVector, Dim>,
                   CurvedScalarWave::Tags::Psi>;
    using compute_items_on_target =
        tmpl::list<CurvedScalarWave::Tags::PsiSquaredCompute,
                   gr::surfaces::Tags::AreaElementCompute<::Frame::Inertial>,
                   gr::surfaces::Tags::SurfaceIntegralCompute<
                       CurvedScalarWave::Tags::PsiSquared, ::Frame::Inertial>>;
    using compute_target_points =
        intrp::TargetPoints::Sphere<SphericalSurface, ::Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<gr::surfaces::Tags::SurfaceIntegralCompute<
                CurvedScalarWave::Tags::PsiSquared, ::Frame::Inertial>>,
            SphericalSurface>>;
    template <typename metavariables>
    using interpolating_component = typename metavariables::dg_element_array;
  };

  using interpolation_target_tags = tmpl::list<SphericalSurface>;
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpatialMetric<DataVector, volume_dim>,
                 CurvedScalarWave::Tags::Psi>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            CurvedScalarWave::BoundaryConditions::BoundaryCondition<volume_dim>,
            CurvedScalarWave::BoundaryConditions::standard_boundary_conditions<
                volume_dim>>,
        tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                dg::Events::field_observations<volume_dim, observe_fields,
                                               non_tensor_compute_tags>,
                tmpl::conditional_t<
                    interpolate,
                    intrp::Events::InterpolateWithoutInterpComponent<
                        volume_dim, SphericalSurface, interpolator_source_vars>,
                    tmpl::list<>>,
                Events::time_events<system>,
                dg::Events::ObserveTimeStepVolume<volume_dim>>>>,
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>,
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::push_back<StepChoosers::standard_step_choosers<system>,
                                   StepChoosers::ByBlock<volume_dim>>>,
        tmpl::pair<StepChooser<StepChooserUse::Slab>,
                   tmpl::push_back<StepChoosers::standard_slab_choosers<
                                       system, local_time_stepping>,
                                   StepChoosers::ByBlock<volume_dim>>>,
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

  using step_actions = tmpl::flatten<tmpl::list<
      CurvedScalarWave::Actions::CalculateGrVars<system, true>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, AllStepChoosers, local_time_stepping,
          use_dg_element_collection>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<
                         tmpl::list<evolution::dg::ApplyBoundaryCorrections<
                             local_time_stepping, system, volume_dim, true>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         system, volume_dim, false, use_dg_element_collection>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  system, volume_dim, false, use_dg_element_collection>,
              Actions::RecordTimeStepperData<system>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<system>>>,
      Actions::CleanHistory<system, local_time_stepping>,
      tmpl::conditional_t<
          use_filtering,
          dg::Actions::Filter<Filters::Exponential<0>,
                              tmpl::list<CurvedScalarWave::Tags::Psi,
                                         CurvedScalarWave::Tags::Pi,
                                         CurvedScalarWave::Tags::Phi<Dim>>>,
          tmpl::list<>>>>;

  using const_global_cache_tags = tmpl::list<
      CurvedScalarWave::Tags::BackgroundSpacetime<BackgroundSpacetime>,
      Tags::AnalyticData<InitialData>>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<volume_dim>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      CurvedScalarWave::Actions::CalculateGrVars<system, false>,
      Initialization::Actions::AddSimpleTags<
          CurvedScalarWave::Initialization::InitializeConstraintDampingGammas<
              volume_dim>,
          CurvedScalarWave::Initialization::InitializeEvolvedVariables<
              volume_dim>>,
      Initialization::Actions::AddComputeTags<
          tmpl::flatten<tmpl::list<StepChoosers::step_chooser_compute_tags<
              EvolutionMetavars, local_time_stepping>>>>,
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
              tmpl::list<
                  evolution::Actions::RunEventsAndTriggers<local_time_stepping>,
                  Actions::ChangeSlabSize, step_actions, Actions::AdvanceTime,
                  PhaseControl::Actions::ExecutePhaseChange>>>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, dg_registration_list>>;
  };

  using component_list = tmpl::flatten<
      tmpl::list<observers::Observer<EvolutionMetavars>,
                 observers::ObserverWriter<EvolutionMetavars>,
                 tmpl::conditional_t<interpolate,
                                     intrp::InterpolationTarget<
                                         EvolutionMetavars, SphericalSurface>,
                                     tmpl::list<>>,
                 dg_element_array>>;

  static constexpr Options::String help{
      "Evolve a scalar wave in Dim spatial dimension on a curved background "
      "spacetime."};

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization,
       Parallel::Phase::InitializeTimeStepperHistory, Parallel::Phase::Register,
       Parallel::Phase::Evolve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
