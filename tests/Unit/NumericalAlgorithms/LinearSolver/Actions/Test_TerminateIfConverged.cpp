// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

namespace {

using simple_tags = db::AddSimpleTags<LinearSolver::Tags::HasConverged>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<LinearSolver::Actions::TerminateIfConverged>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Actions.TerminateIfConverged",
                  "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<ElementArray<Metavariables>>;

  const int self_id{0};

  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};

  SECTION("ProceedIfNotConverged") {
    tuples::get<MockDistributedObjectsTag>(dist_objects)
        .emplace(self_id,
                 db::create<simple_tags>(
                     db::item_type<LinearSolver::Tags::HasConverged>{}));
    MockRuntimeSystem runner{{}, std::move(dist_objects)};
    const auto get_box = [&runner, &self_id]() -> decltype(auto) {
      return runner.algorithms<ElementArray<Metavariables>>()
          .at(self_id)
          .get_databox<ElementArray<Metavariables>::initial_databox>();
    };
    CHECK_FALSE(db::get<LinearSolver::Tags::HasConverged>(get_box()));

    // This should do nothing
    runner.next_action<ElementArray<Metavariables>>(self_id);

    CHECK_FALSE(db::get<LinearSolver::Tags::HasConverged>(get_box()));
    CHECK_FALSE(runner.algorithms<ElementArray<Metavariables>>()
                    .at(self_id)
                    .get_terminate());
  }
  SECTION("TerminateIfConverged") {
    tuples::get<MockDistributedObjectsTag>(dist_objects)
        .emplace(self_id, db::create<simple_tags>(
                              db::item_type<LinearSolver::Tags::HasConverged>{
                                  {1, 0., 0.}, 1, 0., 0.}));
    MockRuntimeSystem runner{{}, std::move(dist_objects)};
    const auto get_box = [&runner, &self_id]() -> decltype(auto) {
      return runner.algorithms<ElementArray<Metavariables>>()
          .at(self_id)
          .get_databox<ElementArray<Metavariables>::initial_databox>();
    };
    CHECK(db::get<LinearSolver::Tags::HasConverged>(get_box()));

    // This should terminate the algorithm
    runner.next_action<ElementArray<Metavariables>>(self_id);

    CHECK(db::get<LinearSolver::Tags::HasConverged>(get_box()));
    CHECK(runner.algorithms<ElementArray<Metavariables>>()
              .at(self_id)
              .get_terminate());
  }
}
