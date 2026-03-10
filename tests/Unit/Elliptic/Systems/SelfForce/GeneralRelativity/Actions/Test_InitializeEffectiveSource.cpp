// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Actions/InitializeEffectiveSource.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::Actions {

namespace {

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<2>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<2>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<tmpl::list<
                  domain::Tags::InitialRefinementLevels<2>,
                  domain::Tags::InitialExtents<2>,
                  ::Tags::Mortars<domain::Tags::Coordinates<2, Frame::Inertial>,
                                  2>>>,
              elliptic::dg::Actions::InitializeDomain<2>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<InitializeEffectiveSource<
                                 GrSelfForce::FirstOrderSystem,
                                 elliptic::Tags::Background<
                                     elliptic::analytic_data::Background>>>>>;
};

struct Metavariables {
  using system = GrSelfForce::FirstOrderSystem;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<elliptic::analytic_data::Background,
                   tmpl::list<GrSelfForce::AnalyticData::CircularOrbit>>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.GrSelfForce.Actions.InitializeEffectiveSource",
                  "[Unit][ParallelAlgorithms]") {
  using background_tag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;
  using element_array = ElementArray<Metavariables>;
  using field_tag = GrSelfForce::Tags::MMode;

  domain::creators::register_derived_with_charm();
  register_factory_classes_with_charm<Metavariables>();
  const domain::creators::AlignedLattice<2> domain_creator{
      {{{-20., 0., 20.}, {0., M_PI}}}, {}, {}, {{2, 2}}, {{4, 4}}, {}, {}, {}};
  const auto element_regularized =
      ElementId<2>{1, {{SegmentId{2, 0}, SegmentId{2, 1}}}};
  const auto element_unregularized =
      ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{2, 1}}}};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      tuples::TaggedTuple<background_tag, domain::Tags::Domain<2>,
                          domain::Tags::FunctionsOfTimeInitialize,
                          elliptic::dg::Tags::Massive,
                          elliptic::dg::Tags::Quadrature>{
          std::make_unique<GrSelfForce::AnalyticData::CircularOrbit>(1.0, 0.0,
                                                                     6.0, 1),
          domain_creator.create_domain(), domain_creator.functions_of_time(),
          false, Spectral::Quadrature::GaussLobatto}};
  const ::Tags::Mortars<domain::Tags::Coordinates<2, Frame::Inertial>, 2>::type
      empty_mortar_coords{};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_regularized,
      {domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents(), empty_mortar_coords});
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_unregularized,
      {domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents(), empty_mortar_coords});
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            element_regularized);
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            element_unregularized);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            element_regularized);
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            element_unregularized);

  const auto get_tag = [&runner](const auto& element_id,
                                 auto tag_v) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };
  const auto check_is_zero = [](const auto& components) {
    return alg::all_of(components, [](const auto& value) {
      return alg::all_of(value, [](const auto& v) { return v == 0.; });
    });
  };

  CHECK(get_tag(element_regularized, Tags::FieldIsRegularized{}) == true);
  // Values of the effective source and singular field are tested in
  // `Test_CircularOrbit.cpp`.
  CHECK_FALSE(check_is_zero(
      get_tag(element_regularized, ::Tags::FixedSource<field_tag>{})));
  CHECK_FALSE(
      check_is_zero(get_tag(element_regularized, Tags::SingularField{})));

  CHECK(get_tag(element_unregularized, Tags::FieldIsRegularized{}) == false);
  CHECK(check_is_zero(
      get_tag(element_unregularized, ::Tags::FixedSource<field_tag>{})));
  CHECK(check_is_zero(get_tag(element_unregularized, Tags::SingularField{})));
}

}  // namespace GrSelfForce::Actions
