// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Evolution/Executables/GrMhd/GhValenciaDivClean/GhValenciaDivCleanBase.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "NumericalAlgorithms/Strahlkorper/IO/InitialShapeFromFile.hpp"
#include "NumericalAlgorithms/Strahlkorper/InitialShape.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FailedHorizonFind.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveFieldsOnHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/ObserveTimeSeriesOnHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Factory.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Events/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/KerrSchild.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

template <bool UseControlSystems, bool UseParametrizedDeleptonization,
          typename... InterpolationTargetTags>
struct EvolutionMetavars
    : public GhValenciaDivCleanTemplateBase<
          EvolutionMetavars<UseControlSystems, UseParametrizedDeleptonization,
                            InterpolationTargetTags...>,
          false, false, UseParametrizedDeleptonization> {
  static_assert(not UseControlSystems,
                "GhValenciaWithHorizon doesn't support control systems yet.");
  static constexpr bool use_dg_subcell = false;

  using defaults = GhValenciaDivCleanDefaults<use_dg_subcell>;
  using base = GhValenciaDivCleanTemplateBase<EvolutionMetavars, use_dg_subcell,
                                              UseControlSystems,
                                              UseParametrizedDeleptonization>;
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
  using initialize_initial_data_dependent_quantities_actions =
      typename defaults::initialize_initial_data_dependent_quantities_actions;

  static constexpr Options::String help{
      "Evolve the Valencia formulation of the GRMHD system with divergence "
      "cleaning, coupled to a dynamic spacetime evolved with the Generalized "
      "Harmonic formulation\n"
      "on a domain with a single horizon and corresponding excised region"};

  struct AhA : tt::ConformsTo<ah::protocols::HorizonMetavars> {
    using time_tag = ah::Tags::ObservationTime<0>;

    using frame = domain_frame;

    using horizon_find_callbacks =
        tmpl::list<ah::callbacks::ObserveTimeSeriesOnHorizon<
            ::ah::tags_for_observing<domain_frame>, AhA>>;
    using horizon_find_failure_callbacks =
        tmpl::list<ah::callbacks::FailedHorizonFind<AhA, false>>;

    using compute_tags_on_element =
        tmpl::list<ah::Tags::ObservationTimeCompute<0>>;

    static constexpr ah::Destination destination = ah::Destination::Observation;

    static std::string name() { return "AhA"; }
  };

  using interpolation_target_tags = tmpl::list<InterpolationTargetTags...>;

  using observe_fields = typename base::observe_fields;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename base::factory_creation::factory_classes,
        tmpl::pair<Event, tmpl::list<ah::Events::FindApparentHorizon<AhA>>>,
        tmpl::pair<ah::Criterion, ah::Criteria::standard_criteria>,
        tmpl::pair<ylm::InitialShape<domain_frame>,
                   tmpl::list<ylm::InitialShapes::Sphere<domain_frame>,
                              ylm::InitialShapes::FromFile<domain_frame>,
                              ah::InitialShapes::KerrSchild<domain_frame>>>>;
  };

  using initial_data_tag = typename base::initial_data_tag;

  using const_global_cache_tags = tmpl::flatten<tmpl::list<
      grmhd::ValenciaDivClean::Tags::PrimitiveFromConservativeOptions,
      gh::gauges::Tags::GaugeCondition, initial_data_tag,
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter,
      typename base::equation_of_state_tag,
      gh::Tags::DampingFunctionGamma0<volume_dim, Frame::Grid>,
      gh::Tags::DampingFunctionGamma1<volume_dim, Frame::Grid>,
      gh::Tags::DampingFunctionGamma2<volume_dim, Frame::Grid>,
      ah::Tags::LMax>>;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::at<typename factory_creation::factory_classes, Event>>;

  using registration = typename base::registration;

  using component_list = tmpl::push_back<typename base::component_list,
                                         ah::Component<EvolutionMetavars, AhA>>;
};
