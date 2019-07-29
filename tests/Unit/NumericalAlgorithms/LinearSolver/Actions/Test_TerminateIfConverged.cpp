// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {

using simple_tags = db::AddSimpleTags<LinearSolver::Tags::HasConverged>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<LinearSolver::Actions::TerminateIfConverged>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Actions.TerminateIfConverged",
                  "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using component = ElementArray<Metavariables>;
  const int self_id{0};

  {
    INFO("ProceedIfNotConverged");
    MockRuntimeSystem runner{{}};
    ActionTesting::emplace_component_and_initialize<component>(
        &runner, self_id, {db::item_type<LinearSolver::Tags::HasConverged>{}});
    runner.set_phase(Metavariables::Phase::Testing);

    CHECK_FALSE(ActionTesting::get_databox_tag<
                component, LinearSolver::Tags::HasConverged>(runner, self_id));

    // This should do nothing
    runner.next_action<component>(self_id);

    CHECK_FALSE(ActionTesting::get_databox_tag<
                component, LinearSolver::Tags::HasConverged>(runner, self_id));
    CHECK_FALSE(ActionTesting::get_terminate<component>(runner, self_id));
  }
  {
    INFO("TerminateIfConverged");
    MockRuntimeSystem runner{{}};
    ActionTesting::emplace_component_and_initialize<component>(
        &runner, self_id,
        {db::item_type<LinearSolver::Tags::HasConverged>{
            {1, 0., 0.}, 1, 0., 0.}});
    runner.set_phase(Metavariables::Phase::Testing);

    CHECK(ActionTesting::get_databox_tag<component,
                                         LinearSolver::Tags::HasConverged>(
        runner, self_id));

    // This should terminate the algorithm
    runner.next_action<component>(self_id);

    CHECK(ActionTesting::get_databox_tag<component,
                                         LinearSolver::Tags::HasConverged>(
        runner, self_id));
    CHECK(ActionTesting::get_terminate<component>(runner, self_id));
  }
}
