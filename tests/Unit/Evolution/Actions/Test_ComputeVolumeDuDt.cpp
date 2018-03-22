// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

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

struct Metavariables;
using component = ActionTesting::MockArrayComponent<
    Metavariables, ElementIndexType, tmpl::list<>,
    tmpl::list<Actions::ComputeVolumeDuDt<2>>>;

struct Metavariables {
  using component_list = tmpl::list<component>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeDuDt",
                  "[Unit][Evolution][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};
  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});
  auto start_box =
      db::create<db::AddSimpleTags<var_tag, Tags::dt<var_tag>>>(3, -100);

  CHECK(db::get<var_tag>(start_box) == 3);
  CHECK(db::get<Tags::dt<var_tag>>(start_box) == -100);
  runner.apply<component, Actions::ComputeVolumeDuDt<2>>(
      start_box, ElementIndexType(self_id));
  CHECK(db::get<var_tag>(start_box) == 3);
  CHECK(db::get<Tags::dt<var_tag>>(start_box) == 6);
}
