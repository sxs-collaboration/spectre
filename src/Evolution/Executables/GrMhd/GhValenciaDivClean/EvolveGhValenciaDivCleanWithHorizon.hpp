// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Executables/GrMhd/GhValenciaDivClean/GhValenciaDivCleanBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeHorizonVolumeQuantities.tpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolationTarget.hpp"
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
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

template <bool UseControlSystems, typename... InterpolationTargetTags>
struct EvolutionMetavars
    : public GhValenciaDivCleanTemplateBase<
          EvolutionMetavars<UseControlSystems, InterpolationTargetTags...>,
          false, false> {
  static_assert(not UseControlSystems,
                "GhValenciaWithHorizon doesn't support control systems yet.");
  static constexpr bool use_dg_subcell = false;

  using defaults = GhValenciaDivCleanDefaults<use_dg_subcell>;
  using base = GhValenciaDivCleanTemplateBase<EvolutionMetavars, use_dg_subcell,
                                              UseControlSystems>;
  static constexpr size_t volume_dim = defaults::volume_dim;
  using domain_frame = typename defaults::domain_frame;
  static constexpr bool use_damped_harmonic_rollon =
      defaults::use_damped_harmonic_rollon;
  using temporal_id = typename defaults::temporal_id;
  static constexpr bool local_time_stepping = defaults::local_time_stepping;
  using system = typename defaults::system;
  using analytic_variables_tags = typename defaults::analytic_variables_tags;
  using analytic_solution_fields = typename defaults::analytic_solution_fields;
  using ordered_list_of_primitive_recovery_schemes =
      typename defaults::ordered_list_of_primitive_recovery_schemes;
  using limiter = typename defaults::limiter;
  using initialize_initial_data_dependent_quantities_actions =
      typename defaults::initialize_initial_data_dependent_quantities_actions;

  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning, coupled to a dynamic spacetime evolved with the Generalized "
      "Harmonic formulation\n"
      "on a domain with a single horizon and corresponding excised region"};

  struct AhA : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using tags_to_observe =
        tmpl::list<gr::surfaces::Tags::AreaCompute<domain_frame>>;
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using vars_to_interpolate_to_target = tmpl::list<
        gr::Tags::SpatialMetric<DataVector, volume_dim, domain_frame>,
        gr::Tags::InverseSpatialMetric<DataVector, volume_dim, domain_frame>,
        gr::Tags::ExtrinsicCurvature<DataVector, volume_dim, domain_frame>,
        gr::Tags::SpatialChristoffelSecondKind<DataVector, volume_dim,
                                               domain_frame>>;
    using compute_items_on_target = tmpl::append<
        tmpl::list<gr::surfaces::Tags::AreaElementCompute<domain_frame>>,
        tags_to_observe>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhA, ::Frame::Inertial>;
    using post_interpolation_callbacks = tmpl::list<
        intrp::callbacks::FindApparentHorizon<AhA, ::Frame::Inertial>>;
    using horizon_find_failure_callback =
        intrp::callbacks::ErrorOnFailedApparentHorizon;
    using post_horizon_find_callbacks = tmpl::list<
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, AhA>>;
  };

  using interpolation_target_tags = tmpl::list<InterpolationTargetTags..., AhA>;
  using interpolator_source_vars = tmpl::list<
      gr::Tags::SpacetimeMetric<DataVector, volume_dim, domain_frame>,
      gh::Tags::Pi<DataVector, volume_dim, domain_frame>,
      gh::Tags::Phi<DataVector, volume_dim, domain_frame>>;

  using observe_fields = typename base::observe_fields;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename base::factory_creation::factory_classes,
        tmpl::pair<Event, tmpl::list<intrp::Events::Interpolate<
                              3, AhA, interpolator_source_vars>>>>;
  };

  using initial_data_tag = typename base::initial_data_tag;

  using const_global_cache_tags = tmpl::flatten<tmpl::list<
      grmhd::ValenciaDivClean::Tags::PrimitiveFromConservativeOptions,
      gh::gauges::Tags::GaugeCondition, initial_data_tag,
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter,
      typename base::equation_of_state_tag,
      gh::ConstraintDamping::Tags::DampingFunctionGamma0<volume_dim,
                                                         Frame::Grid>,
      gh::ConstraintDamping::Tags::DampingFunctionGamma1<volume_dim,
                                                         Frame::Grid>,
      gh::ConstraintDamping::Tags::DampingFunctionGamma2<volume_dim,
                                                         Frame::Grid>>>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  using registration = typename base::registration;

  using component_list =
      tmpl::push_back<typename base::component_list,
                      intrp::InterpolationTarget<EvolutionMetavars, AhA>>;
};
