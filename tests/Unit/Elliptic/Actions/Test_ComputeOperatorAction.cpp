// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Elliptic/Actions/ComputeOperatorAction.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct TemporalId {
  template <typename Tag>
  using step_prefix = LinearSolver::Tags::OperatorAppliedTo<Tag>;
};

struct var_tag : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "var_tag"; }
};

struct ComputeVolumeAx {
  using argument_tags = tmpl::list<var_tag>;
  static void apply(const gsl::not_null<int*> Ax_var, const int& var) {
    *Ax_var = var * 2;
  }
};

struct System {
  using variables_tag = var_tag;
  using compute_operator_action = ComputeVolumeAx;
};

using ElementIndexType = ElementIndex<2>;

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndexType;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using simple_tags =
      db::AddSimpleTags<var_tag,
                        LinearSolver::Tags::OperatorAppliedTo<var_tag>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::Actions::ComputeOperatorAction>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
  using temporal_id = TemporalId;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.ComputeOperatorAction",
                  "[Unit][Elliptic][Actions]") {
  using component = Component<Metavariables>;
  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<component>(&runner, self_id,
                                                             {3, -100});
  runner.set_phase(Metavariables::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<component, var_tag>(runner, self_id) ==
        3);
  CHECK(ActionTesting::get_databox_tag<
            component, LinearSolver::Tags::OperatorAppliedTo<var_tag>>(
            runner, self_id) == -100);
  runner.next_action<component>(self_id);
  CHECK(ActionTesting::get_databox_tag<component, var_tag>(runner, self_id) ==
        3);
  CHECK(ActionTesting::get_databox_tag<
            component, LinearSolver::Tags::OperatorAppliedTo<var_tag>>(
            runner, self_id) == 6);
}
