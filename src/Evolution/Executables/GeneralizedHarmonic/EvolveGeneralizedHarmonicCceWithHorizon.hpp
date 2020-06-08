// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.tpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/Cce/Actions/InterpolateDuringSelfStart.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/RegisterInitializeJWithCharm.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/SendGhWorldtubeData.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

template <typename InitialData, typename BoundaryConditions>
struct EvolutionMetavars
    : public GeneralizedHarmonicTemplateBase<
          EvolutionMetavars<InitialData, BoundaryConditions>>,
      public virtual GeneralizedHarmonicDefaults,
      public CharacteristicExtractDefaults {
  static constexpr bool local_time_stepping = true;
  using cce_boundary_component = Cce::GhWorldtubeBoundary<EvolutionMetavars>;
  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;

  template <bool self_start>
  struct CceWorldtubeTarget;

  using system = typename GeneralizedHarmonicDefaults::system;

  struct AhA {
    using temporal_id = ::Tags::Time;
    using tags_to_observe =
        tmpl::list<StrahlkorperGr::Tags::AreaCompute<frame>,
                   StrahlkorperGr::Tags::IrreducibleMassCompute<frame>>;
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                   gr::Tags::InverseSpatialMetric<volume_dim, frame>,
                   gr::Tags::ExtrinsicCurvature<volume_dim, frame>,
                   gr::Tags::SpatialChristoffelSecondKind<volume_dim, frame>>;
    using compute_items_on_target = tmpl::append<
        tmpl::list<StrahlkorperGr::Tags::AreaElementCompute<frame>,
                   StrahlkorperTags::ThetaPhiCompute<frame>,
                   StrahlkorperTags::RadiusCompute<frame>,
                   StrahlkorperTags::RhatCompute<frame>,
                   StrahlkorperTags::InvJacobianCompute<frame>,
                   StrahlkorperTags::DxRadiusCompute<frame>,
                   StrahlkorperTags::OneOverOneFormMagnitudeCompute<
                       volume_dim, frame, DataVector>,
                   StrahlkorperTags::NormalOneFormCompute<frame>,
                   StrahlkorperTags::UnitNormalOneFormCompute<frame>,
                   StrahlkorperTags::UnitNormalVectorCompute<frame>,
                   StrahlkorperTags::GradUnitNormalOneFormCompute<frame>,
                   StrahlkorperTags::ExtrinsicCurvatureCompute<frame>,
                   StrahlkorperGr::Tags::SpinFunctionCompute<frame>>,
        tags_to_observe>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhA, ::Frame::Inertial>;
    using horizon_find_failure_callback =
        intrp::callbacks::ErrorOnFailedApparentHorizon;
    using post_horizon_find_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, AhA, AhA>;
  };

  using interpolator_source_vars =
      tmpl::list<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>,
                 ::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                 ::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>;

  using dg_registration_list =
      tmpl::push_back<typename GeneralizedHarmonicTemplateBase<
                          EvolutionMetavars>::dg_registration_list,
                      intrp::Actions::RegisterElementWithInterpolator>;

  // use the default same step actions except for interpolating to CCE during
  // self start
  template <bool self_start>
  using step_actions = tmpl::list<
      tmpl::conditional_t<
          self_start,
          Cce::Actions::InterpolateDuringSelfStart<CceWorldtubeTarget<true>>,
          tmpl::list<>>,
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<>,
                     evolution::dg::Actions::ApplyBoundaryCorrections<
                         EvolutionMetavars>>,
          tmpl::list<evolution::dg::Actions::ApplyBoundaryCorrections<
                         EvolutionMetavars>,
                     Actions::RecordTimeStepperData<>,
                     evolution::Actions::RunEventsAndDenseTriggers<>,
                     Actions::UpdateU<>>>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename GeneralizedHarmonicTemplateBase<
            EvolutionMetavars>::factory_creation::factory_classes,
        tmpl::pair<
            Event,
            tmpl::list<
                intrp::Events::Interpolate<3, AhA, interpolator_source_vars>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    volume_dim, CceWorldtubeTarget<true>, EvolutionMetavars,
                    typename CceWorldtubeTarget<
                        true>::vars_to_interpolate_to_target>,
                intrp::Events::Interpolate<
                    volume_dim, CceWorldtubeTarget<false>,
                    typename CceWorldtubeTarget<
                        false>::vars_to_interpolate_to_target>>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, cce_step_choosers>>;
  };
  // the subset of step choosers that are valid for the DG system only
  using dg_step_choosers =
      tmpl::at<typename GeneralizedHarmonicTemplateBase<
                   EvolutionMetavars>::factory_creation::factory_classes,
               StepChooser<StepChooserUse::LtsStep>>;

  using typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars>::analytic_solution_tag;
  using typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars>::analytic_solution_fields;
  using typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars>::phase_changes;

  // initialization actions are the same as the default, with the single
  // addition of initializing the interpolation points (second-to-last action).
  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<volume_dim,
                                            override_functions_of_time>,
      Initialization::Actions::NonconservativeSystem<system>,
      std::conditional_t<
          evolution::is_numeric_initial_data_v<InitialData>, tmpl::list<>,
          evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<volume_dim, Frame::Logical>>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      Initialization::Actions::AddComputeTags<tmpl::push_back<
          StepChoosers::step_chooser_compute_tags<
              GeneralizedHarmonicTemplateBase<EvolutionMetavars>>,
          evolution::Tags::AnalyticCompute<volume_dim, analytic_solution_tag,
                                           analytic_solution_fields>>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using gh_dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::flatten<tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,
          tmpl::conditional_t<
              evolution::is_numeric_initial_data_v<InitialData>,
              tmpl::list<
                  Parallel::PhaseActions<
                      Phase, Phase::RegisterWithElementDataReader,
                      tmpl::list<
                          importers::Actions::RegisterWithElementDataReader,
                          Parallel::Actions::TerminatePhase>>,
                  Parallel::PhaseActions<
                      Phase, Phase::ImportInitialData,
                      tmpl::list<importers::Actions::ReadVolumeData<
                                     evolution::OptionTags::NumericInitialData,
                                     typename system::variables_tag::tags_list>,
                                 importers::Actions::ReceiveVolumeData<
                                     evolution::OptionTags::NumericInitialData,
                                     typename system::variables_tag::tags_list>,
                                 Parallel::Actions::TerminatePhase>>>,
              tmpl::list<>>,
          Parallel::PhaseActions<Phase, Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Phase, Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
          Parallel::PhaseActions<
              Phase, Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions<true>, system>>,
          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<
                  Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                  step_actions<false>, Actions::AdvanceTime,
                  PhaseControl::Actions::ExecutePhaseChange<phase_changes>>>>>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, gh_dg_element_array>,
        dg_registration_list, tmpl::list<>>;
  };

  template <bool self_start>
  struct CceWorldtubeTarget {
    using temporal_id =
        tmpl::conditional_t<self_start, ::Tags::TimeStepId, ::Tags::Time>;

    static std::string name() {
      return self_start ? "SelfStartCceWorldtubeTarget" : "CceWorldtubeTarget";
    }
    using compute_items_on_source = tmpl::list<>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::TargetPoints::KerrHorizon<CceWorldtubeTarget, ::Frame::Inertial>;
    using post_interpolation_callback = intrp::callbacks::SendGhWorldtubeData<
        Cce::CharacteristicEvolution<EvolutionMetavars>, temporal_id,
        self_start>;
    using vars_to_interpolate_to_target =
        tmpl::list<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>,
                   ::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                   ::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>;
    using interpolating_component = gh_dg_element_array;
  };

  using interpolation_target_tags =
      tmpl::list<AhA, CceWorldtubeTarget<true>, CceWorldtubeTarget<false>>;
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          typename AhA::post_horizon_find_callback>>;

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      std::conditional_t<evolution::is_numeric_initial_data_v<InitialData>,
                         importers::ElementDataReader<EvolutionMetavars>,
                         tmpl::list<>>,
      gh_dg_element_array, intrp::Interpolator<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, CceWorldtubeTarget<true>>,
      intrp::InterpolationTarget<EvolutionMetavars, CceWorldtubeTarget<false>>,
      intrp::InterpolationTarget<EvolutionMetavars, AhA>,
      cce_boundary_component, Cce::CharacteristicEvolution<EvolutionMetavars>>>;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation\n"
      "on a domain with a single horizon and corresponding excised region,\n"
      "with a coupled CCE evolution for asymptotic wave data output"};
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryConditions::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryCorrections::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Cce::register_initialize_j_with_charm<
        metavariables::uses_partially_flat_cartesian_coordinates>,
    &Parallel::register_derived_classes_with_charm<Cce::WorldtubeDataManager>,
    &Parallel::register_derived_classes_with_charm<intrp::SpanInterpolator>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
