// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <utility>
// IWYU pragma: no_include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare db::DataBox

namespace {
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
  using du_dt = ComputeDuDt;
};

using ElementIndexType = ElementIndex<2>;

template <typename Metavariables>
struct component : ActionTesting::MockArrayComponent<
                       Metavariables, ElementIndexType, tmpl::list<>,
                       tmpl::list<Actions::ComputeVolumeDuDt<2>>> {
  using initial_databox =
      db::compute_databox_type<tmpl::list<var_tag, Tags::dt<var_tag>>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeDuDt",
                  "[Unit][Evolution][Actions]") {
  using ActionRunner = ActionTesting::ActionRunner<Metavariables>;
  using LocalAlgsTag =
      ActionRunner::LocalAlgorithmsTag<component<Metavariables>>;

  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});

  using simple_tags = db::AddSimpleTags<var_tag, Tags::dt<var_tag>>;
  ActionRunner::LocalAlgorithms local_algs{};
  tuples::get<LocalAlgsTag>(local_algs)
      .emplace(self_id, db::create<simple_tags>(3, -100));
  ActionRunner runner{{}, std::move(local_algs)};
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
