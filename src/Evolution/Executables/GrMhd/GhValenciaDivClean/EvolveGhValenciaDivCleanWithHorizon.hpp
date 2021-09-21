// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.tpp"
#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Executables/GrMhd/GhValenciaDivClean/GhValenciaDivCleanBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
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
#include "Time/StepControllers/Factory.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/TMPL.hpp"

template <typename InitialData, typename... InterpolationTargetTags>
struct EvolutionMetavars
    : public virtual GhValenciaDivCleanDefaults,
      public GhValenciaDivCleanTemplateBase<
          EvolutionMetavars<InitialData, InterpolationTargetTags...>> {
  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning, coupled to a dynamic spacetime evolved with the Generalized "
      "Harmonic formulation\n"
      "on a domain with a single horizon and corresponding excised region"};

  struct AhA {
    using temporal_id = ::Tags::Time;
    using tags_to_observe =
        tmpl::list<StrahlkorperGr::Tags::AreaCompute<domain_frame>>;
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using vars_to_interpolate_to_target = tmpl::list<
        gr::Tags::SpatialMetric<volume_dim, domain_frame, DataVector>,
        gr::Tags::InverseSpatialMetric<volume_dim, domain_frame>,
        gr::Tags::ExtrinsicCurvature<volume_dim, domain_frame>,
        gr::Tags::SpatialChristoffelSecondKind<volume_dim, domain_frame>>;
    using compute_items_on_target = tmpl::append<
        tmpl::list<StrahlkorperGr::Tags::AreaElementCompute<domain_frame>>,
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
      tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, domain_frame>,
                 GeneralizedHarmonic::Tags::Pi<volume_dim, domain_frame>,
                 GeneralizedHarmonic::Tags::Phi<volume_dim, domain_frame>>;

  using observe_fields = typename GhValenciaDivCleanTemplateBase<
      EvolutionMetavars>::observe_fields;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename GhValenciaDivCleanTemplateBase<
            EvolutionMetavars>::factory_creation::factory_classes,
        tmpl::pair<Event, tmpl::list<intrp::Events::Interpolate<
                              3, AhA, interpolator_source_vars>>>>;
  };

  using initial_data =
      typename GhValenciaDivCleanTemplateBase<EvolutionMetavars>::initial_data;
  using initial_data_tag = typename GhValenciaDivCleanTemplateBase<
      EvolutionMetavars>::initial_data_tag;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          typename AhA::post_horizon_find_callback,
          typename InterpolationTargetTags::post_interpolation_callback...>>;

  using dg_registration_list = typename GhValenciaDivCleanTemplateBase<
      EvolutionMetavars>::dg_registration_list;

  using phase_selection =
      typename GhValenciaDivCleanTemplateBase<EvolutionMetavars>::
          template PhaseSelection<
              EvolutionMetavars,
              typename GhValenciaDivCleanTemplateBase<
                  EvolutionMetavars>::dg_element_array_component,
              typename GhValenciaDivCleanTemplateBase<
                  EvolutionMetavars>::dg_registration_list>;

  using const_global_cache_tags = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<evolution::is_numeric_initial_data_v<InitialData>,
                          tmpl::list<>, initial_data_tag>,
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter,
      Tags::EventsAndTriggers,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma0<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma1<
          volume_dim, Frame::Grid>,
      GeneralizedHarmonic::ConstraintDamping::Tags::DampingFunctionGamma2<
          volume_dim, Frame::Grid>,
      PhaseControl::Tags::PhaseChangeAndTriggers<
          typename phase_selection::phase_changes>>>;

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

  using component_list =
      tmpl::push_back<typename GhValenciaDivCleanTemplateBase<
                          EvolutionMetavars>::component_list,
                      intrp::InterpolationTarget<EvolutionMetavars, AhA>>;
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &grmhd::GhValenciaDivClean::BoundaryCorrections::
        register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        PhaseChange<metavariables::phase_selection::phase_changes>>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
