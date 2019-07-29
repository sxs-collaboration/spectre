// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "ErrorHandling/Error.hpp"
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/ConstGlobalCache.hpp"
#include "ParallelBackend/Invoke.hpp"
#include "ParallelBackend/NodeLock.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

// IWYU pragma: no_include <lrtslock.h>

namespace {
namespace TestSimpleAndThreadedActions {
struct simple_action_a;
struct simple_action_a_mock;
struct simple_action_c;
struct simple_action_c_mock;

struct threaded_action_b;
struct threaded_action_b_mock;

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
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<ValueTag, PassedToB>>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<simple_action_a, simple_action_c, threaded_action_b>;
  using with_these_simple_actions =
      tmpl::list<simple_action_a_mock, simple_action_c_mock,
                 threaded_action_b_mock>;
  using replace_these_threaded_actions = tmpl::list<threaded_action_b>;
  using with_these_threaded_actions = tmpl::list<threaded_action_b_mock>;
};

struct simple_action_a_mock {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const int value) noexcept {
    db::mutate<ValueTag>(
        make_not_null(&box), [&value](
                                 const gsl::not_null<int*> value_box) noexcept {
          *value_box = value;
        });
  }
};

struct simple_action_b {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, PassedToB>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,                      // NOLINT
                    Parallel::ConstGlobalCache<Metavariables>& cache,  // NOLINT
                    const ArrayIndex& /*array_index*/,
                    const int to_call) noexcept {
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
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) noexcept {
    db::mutate<ValueTag>(
        make_not_null(&box), [](const gsl::not_null<int*> value_box) noexcept {
          *value_box = 25;
        });
  }
};

struct threaded_action_a {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<CmiNodeLock*> /*node_lock*/) noexcept {
    db::mutate<ValueTag>(make_not_null(&box), [
    ](const gsl::not_null<int*> value_box) noexcept { *value_box = 35; });
  }
};

struct threaded_action_b_mock {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<CmiNodeLock*> node_lock,
                    const int tag) noexcept {
    Parallel::lock(node_lock);
    db::mutate<ValueTag>(
        make_not_null(&box), [tag](
                                 const gsl::not_null<int*> value_box) noexcept {
          *value_box = tag;
        });
    Parallel::unlock(node_lock);
  }
};

struct SimpleActionMockMetavariables {
  using component_list = tmpl::list<
      component_for_simple_action_mock<SimpleActionMockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.ActionTesting.MockSimpleAction", "[Unit]") {
  using metavars = SimpleActionMockMetavariables;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component_and_initialize<
      component_for_simple_action_mock<metavars>>(&runner, 0, {0, -1});
  runner.set_phase(metavars::Phase::Testing);

  const auto& box =
      ActionTesting::get_databox<component_for_simple_action_mock<metavars>,
                                 db::AddSimpleTags<ValueTag, PassedToB>>(runner,
                                                                         0);
  CHECK(db::get<PassedToB>(box) == -1);
  runner.simple_action<component_for_simple_action_mock<metavars>,
                       simple_action_b>(0, 0);
  REQUIRE(not runner.is_simple_action_queue_empty<
              component_for_simple_action_mock<metavars>>(0));
  runner
      .invoke_queued_simple_action<component_for_simple_action_mock<metavars>>(
          0);
  CHECK(db::get<PassedToB>(box) == 0);
  CHECK(db::get<ValueTag>(box) == 11);
  runner.simple_action<component_for_simple_action_mock<metavars>,
                       simple_action_b>(0, 2);
  REQUIRE(not runner.is_simple_action_queue_empty<
              component_for_simple_action_mock<metavars>>(0));
  runner
      .invoke_queued_simple_action<component_for_simple_action_mock<metavars>>(
          0);
  CHECK(db::get<PassedToB>(box) == 2);
  CHECK(db::get<ValueTag>(box) == 25);
  runner.simple_action<component_for_simple_action_mock<metavars>,
                       simple_action_b>(0, 1);
  REQUIRE(not runner.is_simple_action_queue_empty<
              component_for_simple_action_mock<metavars>>(0));
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

  ActionTesting::threaded_action<component_for_simple_action_mock<metavars>,
                                 threaded_action_a>(make_not_null(&runner), 0);
  CHECK(db::get<ValueTag>(box) == 35);

  ActionTesting::threaded_action<component_for_simple_action_mock<metavars>,
                                 threaded_action_b>(make_not_null(&runner), 0,
                                                    -50);
  CHECK(db::get<ValueTag>(box) == -50);
}
}  // namespace TestSimpleAndThreadedActions

namespace TestIsRetrievable {
struct DummyTimeTag : db::SimpleTag {
  static std::string name() noexcept { return "DummyTime"; }
  using type = int;
};

struct Action0 {
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTagsList, DummyTimeTag>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        db::create_from<db::RemoveTags<>, db::AddSimpleTags<DummyTimeTag>>(
            std::move(box), 6));
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTagsList, DummyTimeTag>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box)};
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Testing,
                                        tmpl::list<Action0>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.ActionTesting.IsRetrievable", "[Unit]") {
  using metavars = Metavariables;
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);
  runner.set_phase(Metavariables::Phase::Testing);
  CHECK(not ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner,
                                                                       0));
  // Runs Action0
  runner.next_action<component>(0);
  CHECK(ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner, 0));
  CHECK(ActionTesting::get_databox_tag<component, DummyTimeTag>(runner, 0) ==
        6);
}
}  // namespace TestIsRetrievable
}  // namespace
