// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct TemporalId {
  template <typename Tag>
  using step_prefix = Tags::dt<Tag>;
};

struct var_tag : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "var_tag"; }
};

struct ComputeDuDt {
  using argument_tags = tmpl::list<var_tag>;
  static void apply(const gsl::not_null<int*> dt_var, const int& var) {
    *dt_var = var * 2;
  }
};

struct System {
  using variables_tag = var_tag;
  using compute_time_derivative = ComputeDuDt;
};

using ElementIndexType = ElementIndex<2>;

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndexType;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<Actions::ComputeTimeDerivative>;
  using initial_databox =
      db::compute_databox_type<tmpl::list<var_tag, Tags::dt<var_tag>>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
  using temporal_id = TemporalId;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeTimeDerivative",
                  "[Unit][Evolution][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<component<Metavariables>>;

  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});

  using simple_tags = db::AddSimpleTags<var_tag, Tags::dt<var_tag>>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id, db::create<simple_tags>(3, -100));
  MockRuntimeSystem runner{{}, std::move(dist_objects)};
  const auto get_box = [&runner, &self_id]() -> decltype(auto) {
    return runner.algorithms<component<Metavariables>>()
        .at(self_id)
        .get_databox<db::compute_databox_type<simple_tags>>();
  };
  {
    const auto& box = get_box();
    CHECK(db::get<var_tag>(box) == 3);
    CHECK(db::get<Tags::dt<var_tag>>(box) == -100);
  }
  runner.next_action<component<Metavariables>>(self_id);
  {
    const auto& box = get_box();
    CHECK(db::get<var_tag>(box) == 3);
    CHECK(db::get<Tags::dt<var_tag>>(box) == 6);
  }
}
