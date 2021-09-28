// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeBackgroundFields.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct BackgroundFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using background_fields = tmpl::list<BackgroundFieldTag>;
};

struct Background {
  static tuples::TaggedTuple<BackgroundFieldTag> variables(
      const tnsr::I<DataVector, 1>& x, const Mesh<1>& /*mesh*/,
      const InverseJacobian<DataVector, 1, Frame::ElementLogical,
                            Frame::Inertial>&
      /*inv_jacobian*/,
      tmpl::list<BackgroundFieldTag> /*meta*/) {
    return {Scalar<DataVector>{get<0>(x)}};
  }
  // NOLINTNEXTLINE
  void pup(PUP::er& /*p*/) {}
};

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<1>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<domain::Tags::InitialRefinementLevels<1>,
                                    domain::Tags::InitialExtents<1>>>,
                     Actions::SetupDataBox,
                     elliptic::dg::Actions::InitializeDomain<1>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<::elliptic::Actions::InitializeBackgroundFields<
              typename Metavariables::system,
              elliptic::Tags::Background<Background>>>>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tags =
      tmpl::list<elliptic::Tags::Background<Background>>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeBackgroundFields",
                  "[Unit][Elliptic][Actions]") {
  domain::creators::register_derived_with_charm();
  // Which element we work with does not matter for this test
  const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
  const domain::creators::Interval domain_creator{{{-0.5}}, {{1.5}},   {{2}},
                                                  {{4}},    {{false}}, nullptr};

  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::make_unique<Background>(), domain_creator.create_domain()}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id,
      {domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents()});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  const auto get_tag = [&runner, &element_id ](auto tag_v) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  const auto& inertial_coords =
      get_tag(domain::Tags::Coordinates<1, Frame::Inertial>{});
  CHECK(get(get_tag(BackgroundFieldTag{})) == get<0>(inertial_coords));
}
