// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {

struct DummyOptionsGroup {};

using simple_tags =
    db::AddSimpleTags<LinearSolver::Tags::HasConverged<DummyOptionsGroup>>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              LinearSolver::Actions::TerminateIfConverged<DummyOptionsGroup>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
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
        &runner, self_id, {Convergence::HasConverged{}});
    ActionTesting::set_phase(make_not_null(&runner),
                             Metavariables::Phase::Testing);

    CHECK_FALSE(ActionTesting::get_databox_tag<
                component, LinearSolver::Tags::HasConverged<DummyOptionsGroup>>(
        runner, self_id));

    // This should do nothing
    runner.next_action<component>(self_id);

    CHECK_FALSE(ActionTesting::get_databox_tag<
                component, LinearSolver::Tags::HasConverged<DummyOptionsGroup>>(
        runner, self_id));
    CHECK_FALSE(ActionTesting::get_terminate<component>(runner, self_id));
  }
  {
    INFO("TerminateIfConverged");
    MockRuntimeSystem runner{{}};
    ActionTesting::emplace_component_and_initialize<component>(
        &runner, self_id, {Convergence::HasConverged{{1, 0., 0.}, 1, 0., 0.}});
    ActionTesting::set_phase(make_not_null(&runner),
                             Metavariables::Phase::Testing);

    CHECK(ActionTesting::get_databox_tag<
          component, LinearSolver::Tags::HasConverged<DummyOptionsGroup>>(
        runner, self_id));

    // This should terminate the algorithm
    runner.next_action<component>(self_id);

    CHECK(ActionTesting::get_databox_tag<
          component, LinearSolver::Tags::HasConverged<DummyOptionsGroup>>(
        runner, self_id));
    CHECK(ActionTesting::get_terminate<component>(runner, self_id));
  }
}
