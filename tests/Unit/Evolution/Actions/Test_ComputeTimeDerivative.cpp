// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct TemporalId {
  template <typename Tag>
  using step_prefix = Tags::dt<Tag>;
};

struct var_tag : db::SimpleTag {
  using type = int;
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
  using simple_tags = tmpl::list<var_tag, Tags::dt<var_tag>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::ComputeTimeDerivative>>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  using temporal_id = TemporalId;

  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeTimeDerivative",
                  "[Unit][Evolution][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});

  using simple_tags = db::AddSimpleTags<var_tag, Tags::dt<var_tag>>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component<Metavariables>>(
      &runner, self_id, {3, -100});
  runner.set_phase(Metavariables::Phase::Testing);

  {
    const auto& box =
        ActionTesting::get_databox<component<Metavariables>, simple_tags>(
            runner, self_id);
    CHECK(db::get<var_tag>(box) == 3);
    CHECK(db::get<Tags::dt<var_tag>>(box) == -100);
  }
  runner.next_action<component<Metavariables>>(self_id);
  {
    const auto& box =
        ActionTesting::get_databox<component<Metavariables>, simple_tags>(
            runner, self_id);
    CHECK(db::get<var_tag>(box) == 3);
    CHECK(db::get<Tags::dt<var_tag>>(box) == 6);
  }
}
