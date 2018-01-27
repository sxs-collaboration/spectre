// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeVolumeDuDt.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct var_tag : db::DataBoxTag {
  using type = int;
  static constexpr db::DataBoxString label = "var_tag";
};
struct dt_var_tag : db::DataBoxTag {
  using type = int;
  static constexpr db::DataBoxString label = "dt_var_tag";
};

struct ComputeDuDt {
  using return_tags = tmpl::list<dt_var_tag>;
  using argument_tags = tmpl::list<var_tag>;
  static void apply(const gsl::not_null<int*> dt_var, const int& var) {
    *dt_var = var * 2;
  }
};

struct System {
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
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeVolumeDuDt",
                  "[Unit][Evolution][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};
  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});
  auto start_box = db::create<db::AddTags<var_tag, dt_var_tag>>(3, -100);

  CHECK(db::get<var_tag>(start_box) == 3);
  CHECK(db::get<dt_var_tag>(start_box) == -100);
  runner.apply<component, Actions::ComputeVolumeDuDt<2>>(
      start_box, ElementIndexType(self_id));
  CHECK(db::get<var_tag>(start_box) == 3);
  CHECK(db::get<dt_var_tag>(start_box) == 6);
}
