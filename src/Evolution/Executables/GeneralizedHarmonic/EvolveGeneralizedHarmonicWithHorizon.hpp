// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.tpp"
#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/StepControllers/Factory.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

// First template parameter specifies the source of the initial data, which
// could be an analytic solution, analytic data, or imported numerical data.
// Second template parameter specifies the analytic solution used when imposing
// dirichlet boundary conditions or against which to compute error norms.
template <typename InitialData, typename BoundaryConditions>
struct EvolutionMetavars<3, InitialData, BoundaryConditions>
    : public GeneralizedHarmonicTemplateBase<
          true, EvolutionMetavars<3, InitialData, BoundaryConditions>> {
  using gh_base = GeneralizedHarmonicTemplateBase<
      true, EvolutionMetavars<3, InitialData, BoundaryConditions>>;
  using typename gh_base::frame;
  using typename gh_base::initialize_initial_data_dependent_quantities_actions;
  using typename gh_base::Phase;
  using typename gh_base::system;
  static constexpr size_t volume_dim = 3;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation,\n"
      "on a domain with a single horizon and corresponding excised region"};

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

  using interpolation_target_tags = tmpl::list<AhA>;
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Pi<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Phi<volume_dim, frame>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename gh_base::factory_creation::factory_classes,
        tmpl::pair<Event, tmpl::list<intrp::Events::Interpolate<
                              3, AhA, interpolator_source_vars>>>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          typename AhA::post_horizon_find_callback>>;

  using dg_registration_list =
      tmpl::push_back<typename gh_base::dg_registration_list,
                      intrp::Actions::RegisterElementWithInterpolator>;

  using typename gh_base::initialization_actions;

  using typename gh_base::step_actions;

  // the dg element array needs to be re-declared to capture the new type
  // aliases for the action lists.
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
          Parallel::PhaseActions<
              Phase, Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
          Parallel::PhaseActions<
              Phase, Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
          Parallel::PhaseActions<Phase, Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>>;

  using phase_selection = typename gh_base::template PhaseSelection<
      EvolutionMetavars, typename gh_base::gh_dg_element_array,
      typename gh_base::dg_registration_list>;

  using const_global_cache_tags = tmpl::list<
      typename gh_base::analytic_solution_tag, Tags::EventsAndTriggers,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma0<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma1<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma2<
          volume_dim, Frame::Grid>,
      PhaseControl::Tags::PhaseChangeAndTriggers<
          typename phase_selection::phase_changes>>;

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      std::conditional_t<evolution::is_numeric_initial_data_v<InitialData>,
                         importers::ElementDataReader<EvolutionMetavars>,
                         tmpl::list<>>,
      gh_dg_element_array, intrp::Interpolator<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, AhA>>>;

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<EvolutionMetavars>&
          cache_proxy) noexcept {
    const auto next_phase = PhaseControl::arbitrate_phase_change<
        typename phase_selection::phase_changes>(
        phase_change_decision_data, current_phase,
        *(cache_proxy.ckLocalBranch()));
    if (next_phase.has_value()) {
      return next_phase.value();
    }
    switch (current_phase) {
      case Phase::Initialization:
        return evolution::is_numeric_initial_data_v<InitialData>
                   ? Phase::RegisterWithElementDataReader
                   : Phase::InitializeInitialDataDependentQuantities;
      case Phase::RegisterWithElementDataReader:
        return Phase::ImportInitialData;
      case Phase::ImportInitialData:
        return Phase::InitializeInitialDataDependentQuantities;
      case Phase::InitializeInitialDataDependentQuantities:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::Register;
      case Phase::Register:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryCorrections::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_selection::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
