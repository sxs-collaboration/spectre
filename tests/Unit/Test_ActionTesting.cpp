// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "ErrorHandling/Error.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct simple_action_a;
struct simple_action_a_mock;
struct simple_action_c;
struct simple_action_c_mock;

struct ValueTag : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "ValueTag"; }
};

struct PassedToB : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "PassedToB"; }
};

template <typename Metavariables>
struct component_for_simple_action_mock {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<tmpl::list<ValueTag, PassedToB>>;

  using replace_these_simple_actions =
      tmpl::list<simple_action_a, simple_action_c>;
  using with_these_simple_actions =
      tmpl::list<simple_action_a_mock, simple_action_c_mock>;
};

struct simple_action_a_mock {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<ValueTag, PassedToB>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/, const int value) noexcept {
    db::mutate<ValueTag>(
        make_not_null(&box), [&value](
                                 const gsl::not_null<int*> value_box) noexcept {
          *value_box = value;
        });
  }
};

struct simple_action_b {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<ValueTag, PassedToB>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,  // NOLINT
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/, const int to_call) noexcept {
    // simple_action_b is the action that we are testing, but it calls some
    // other actions that we don't want to test. Those are mocked where the body
    // of the mock actions records something in the DataBox that we check to
    // verify the mock actions were actually called.
    //
    // We do some "work" here by updating the `PassedToB` tag in the DataBox.
    db::mutate<PassedToB>(
        make_not_null(&box), [&to_call](const gsl::not_null<int*>
                                            passed_to_b) noexcept {
          *passed_to_b = to_call;
        });
    if (to_call == 0) {
      Parallel::simple_action<simple_action_a>(
          Parallel::get_parallel_component<
              component_for_simple_action_mock<Metavariables>>(cache),
          11);
    } else if (to_call == 1) {
      Parallel::simple_action<simple_action_a>(
          Parallel::get_parallel_component<
              component_for_simple_action_mock<Metavariables>>(cache),
          14);
    } else if (to_call == 2) {
      Parallel::simple_action<simple_action_c>(
          Parallel::get_parallel_component<
              component_for_simple_action_mock<Metavariables>>(cache));
    } else if (to_call == 3) {
      Parallel::simple_action<simple_action_c>(
          Parallel::get_parallel_component<
              component_for_simple_action_mock<Metavariables>>(cache)[0]);
    } else {
      ERROR("to_call should be 0, 1, 2, or 3");
    }
  }
};

struct simple_action_c_mock {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<ValueTag, PassedToB>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<ValueTag>(
        make_not_null(&box), [](const gsl::not_null<int*> value_box) noexcept {
          *value_box = 25;
        });
  }
};

struct SimpleActionMockMetavariables {
  using component_list = tmpl::list<
      component_for_simple_action_mock<SimpleActionMockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.ActionTesting.MockSimpleAction", "[Unit]") {
  using metavars = SimpleActionMockMetavariables;
  using TupleOfMockDistributedObjects =
      typename ActionTesting::MockRuntimeSystem<
          metavars>::TupleOfMockDistributedObjects;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  TupleOfMockDistributedObjects dist_objects{};
  using LocalAlgTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          component_for_simple_action_mock<metavars>>;
  tuples::get<LocalAlgTag>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<
                   component_for_simple_action_mock<metavars>>{
                   db::create<db::AddSimpleTags<ValueTag, PassedToB>>(0, -1)});
  ActionTesting::MockRuntimeSystem<metavars> runner{{},
                                                    std::move(dist_objects)};
  const auto& box =
      runner.template algorithms<component_for_simple_action_mock<metavars>>()
          .at(0)
          .template get_databox<typename component_for_simple_action_mock<
              metavars>::initial_databox>();
  CHECK(db::get<PassedToB>(box) == -1);
  runner.simple_action<component_for_simple_action_mock<metavars>,
                       simple_action_b>(0, 0);
  runner
      .invoke_queued_simple_action<component_for_simple_action_mock<metavars>>(
          0);
  CHECK(db::get<PassedToB>(box) == 0);
  CHECK(db::get<ValueTag>(box) == 11);
  runner.simple_action<component_for_simple_action_mock<metavars>,
                       simple_action_b>(0, 2);
  runner
      .invoke_queued_simple_action<component_for_simple_action_mock<metavars>>(
          0);
  CHECK(db::get<PassedToB>(box) == 2);
  CHECK(db::get<ValueTag>(box) == 25);
  runner.simple_action<component_for_simple_action_mock<metavars>,
                       simple_action_b>(0, 1);
  runner
      .invoke_queued_simple_action<component_for_simple_action_mock<metavars>>(
          0);
  CHECK(db::get<PassedToB>(box) == 1);
  CHECK(db::get<ValueTag>(box) == 14);
  runner.simple_action<component_for_simple_action_mock<metavars>,
                       simple_action_b>(0, 3);

  CHECK_FALSE(runner.is_simple_action_queue_empty<
              component_for_simple_action_mock<metavars>>(0));
  runner
      .invoke_queued_simple_action<component_for_simple_action_mock<metavars>>(
          0);
  CHECK(runner.is_simple_action_queue_empty<
        component_for_simple_action_mock<metavars>>(0));
  CHECK(db::get<PassedToB>(box) == 3);
  CHECK(db::get<ValueTag>(box) == 25);
}
}  // namespace
