// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using primal_fields = tmpl::list<ScalarFieldTag>;
};

struct Background : elliptic::analytic_data::Background {
  Background() = default;
  explicit Background(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(Background);  // NOLINT
#pragma GCC diagnostic pop

  // NOLINTBEGIN(readability-convert-member-functions-to-static)
  // [background_vars_fct]
  tuples::TaggedTuple<Tags::FixedSource<ScalarFieldTag>> variables(  // NOLINT
      const tnsr::I<DataVector, 1>& x,
      tmpl::list<Tags::FixedSource<ScalarFieldTag>> /*meta*/) const {
    // [background_vars_fct]
    return {Scalar<DataVector>{get<0>(x)}};
  }
  // NOLINTEND(readability-convert-member-functions-to-static)
};

PUP::able::PUP_ID Background::my_PUP_ID = 0;  // NOLINT

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<1>>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
                                    domain::Tags::InitialRefinementLevels<1>,
                                    domain::Tags::InitialExtents<1>>>,
                                elliptic::dg::Actions::InitializeDomain<1>>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Testing,
                     tmpl::list<elliptic::Actions::InitializeFixedSources<
                         typename Metavariables::system,
                         elliptic::Tags::Background<
                             elliptic::analytic_data::Background>>>>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tags = tmpl::list<
      elliptic::Tags::Background<elliptic::analytic_data::Background>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<elliptic::analytic_data::Background,
                             tmpl::list<Background>>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeFixedSources",
                  "[Unit][Elliptic][Actions]") {
  domain::creators::register_derived_with_charm();
  Parallel::register_factory_classes_with_charm<Metavariables>();
  // Which element we work with does not matter for this test
  const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
  const domain::creators::Interval domain_creator{
      {{-0.5}}, {{1.5}}, {{2}}, {{4}}};

  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{tuples::TaggedTuple<
      elliptic::Tags::Background<elliptic::analytic_data::Background>,
      domain::Tags::Domain<1>, elliptic::dg::Tags::Massive>{
      std::make_unique<Background>(), domain_creator.create_domain(), false}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id,
      {domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents()});
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  const auto get_tag = [&runner, &element_id ](auto tag_v) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  const auto& inertial_coords =
      get_tag(domain::Tags::Coordinates<1, Frame::Inertial>{});
  CHECK(get(get_tag(::Tags::FixedSource<ScalarFieldTag>{})) ==
        get<0>(inertial_coords));
}
