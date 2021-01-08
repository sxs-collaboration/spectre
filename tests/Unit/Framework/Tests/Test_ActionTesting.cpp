// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

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

/// [tags for const global cache]
struct ValueTag : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "ValueTag"; }
};

struct PassedToB : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "PassedToB"; }
};
/// [tags for const global cache]

template <typename Metavariables>
struct component_for_simple_action_mock {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<ValueTag, PassedToB>>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  /// [simple action replace]
  using replace_these_simple_actions =
      tmpl::list<simple_action_a, simple_action_c, threaded_action_b>;
  using with_these_simple_actions =
      tmpl::list<simple_action_a_mock, simple_action_c_mock,
                 threaded_action_b_mock>;
  /// [simple action replace]
  /// [threaded action replace]
  using replace_these_threaded_actions = tmpl::list<threaded_action_b>;
  using with_these_threaded_actions = tmpl::list<threaded_action_b_mock>;
  /// [threaded action replace]
};

struct simple_action_a_mock {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
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
                    Parallel::GlobalCache<Metavariables>& cache,  // NOLINT
                    const ArrayIndex& /*array_index*/,
                    const double to_call) noexcept {
    // simple_action_b is the action that we are testing, but it calls some
    // other actions that we don't want to test. Those are mocked where the body
    // of the mock actions records something in the DataBox that we check to
    // verify the mock actions were actually called.
    //
    // We do some "work" here by updating the `PassedToB` tag in the DataBox.
    db::mutate<PassedToB>(
        make_not_null(&box),
        [&to_call](const gsl::not_null<double*> passed_to_b) noexcept {
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
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
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
  static void apply(
      db::DataBox<DbTagsList>& box,  // NOLINT
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const gsl::not_null<Parallel::NodeLock*> /*node_lock*/) noexcept {
    db::mutate<ValueTag>(make_not_null(&box), [
    ](const gsl::not_null<int*> value_box) noexcept { *value_box = 35; });
  }
};

struct threaded_action_b_mock {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const int tag) noexcept {
    node_lock->lock();
    db::mutate<ValueTag>(make_not_null(&box),
                         [tag](const gsl::not_null<int*> value_box) noexcept {
                           *value_box = tag;
                         });
    node_lock->unlock();
  }
};

struct SimpleActionMockMetavariables {
  using component_list = tmpl::list<
      component_for_simple_action_mock<SimpleActionMockMetavariables>>;

  enum class Phase { Initialization, Testing, Exit };
};

struct MockMetavariablesWithGlobalCacheTags {
  using component_list = tmpl::list<
      component_for_simple_action_mock<MockMetavariablesWithGlobalCacheTags>>;
  /// [const global cache metavars]
  using const_global_cache_tags = tmpl::list<ValueTag, PassedToB>;
  /// [const global cache metavars]

  enum class Phase { Initialization, Testing, Exit };
};

void test_mock_runtime_system_constructors() {
  using metavars = MockMetavariablesWithGlobalCacheTags;
  // Test whether we can construct with tagged tuples in different orders.
  /// [constructor const global cache tags known]
  ActionTesting::MockRuntimeSystem<metavars> runner1{{3, 7.0}};
  /// [constructor const global cache tags known]
  /// [constructor const global cache tags unknown]
  ActionTesting::MockRuntimeSystem<metavars> runner2{
      tuples::TaggedTuple<PassedToB, ValueTag>{7.0, 3}};
  /// [constructor const global cache tags unknown]
  CHECK(Parallel::get<ValueTag>(runner1.cache()) ==
        Parallel::get<ValueTag>(runner2.cache()));
  CHECK(Parallel::get<PassedToB>(runner1.cache()) ==
        Parallel::get<PassedToB>(runner2.cache()));
  CHECK(Parallel::get<ValueTag>(runner1.cache()) == 3);
  CHECK(Parallel::get<PassedToB>(runner1.cache()) == 7);
}

SPECTRE_TEST_CASE("Unit.ActionTesting.MockSimpleAction", "[Unit]") {
  using metavars = SimpleActionMockMetavariables;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component_and_initialize<
      component_for_simple_action_mock<metavars>>(&runner, 0, {0, -1});
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  /// [get databox]
  const auto& box =
      ActionTesting::get_databox<component_for_simple_action_mock<metavars>,
                                 db::AddSimpleTags<ValueTag, PassedToB>>(runner,
                                                                         0);
  /// [get databox]
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
  /// [invoke simple action]
  ActionTesting::simple_action<component_for_simple_action_mock<metavars>,
                               simple_action_b>(make_not_null(&runner), 0, 2);
  /// [invoke simple action]
  REQUIRE(not
          /// [simple action queue empty]
          ActionTesting::is_simple_action_queue_empty<
              component_for_simple_action_mock<metavars>>(runner, 0)
          /// [simple action queue empty]
  );
  /// [invoke queued simple action]
  ActionTesting::invoke_queued_simple_action<
      component_for_simple_action_mock<metavars>>(make_not_null(&runner), 0);
  /// [invoke queued simple action]
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

  /// [invoke threaded action]
  ActionTesting::threaded_action<component_for_simple_action_mock<metavars>,
                                 threaded_action_a>(make_not_null(&runner), 0);
  /// [invoke threaded action]
  CHECK(db::get<ValueTag>(box) == 35);

  ActionTesting::threaded_action<component_for_simple_action_mock<metavars>,
                                 threaded_action_b>(make_not_null(&runner), 0,
                                                    -50);
  CHECK(db::get<ValueTag>(box) == -50);

  test_mock_runtime_system_constructors();
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
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
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
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
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

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Testing,
                                        tmpl::list<Action0>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  enum class Phase { Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.ActionTesting.IsRetrievable", "[Unit]") {
  using metavars = Metavariables;
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  CHECK(not ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner,
                                                                       0));
  // Runs Action0
  runner.next_action<component>(0);
  CHECK(
      /// [tag is retrievable]
      ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner, 0)
      /// [tag is retrievable]
  );
  CHECK(ActionTesting::get_databox_tag<component, DummyTimeTag>(runner, 0) ==
        6);
}
}  // namespace TestIsRetrievable

namespace InboxTags {
namespace Tags {
struct ValueTag : public Parallel::InboxInserters::Value<ValueTag> {
  using temporal_id = size_t;
  using type = std::unordered_map<temporal_id, int>;
};
}  // namespace Tags

namespace Actions {
struct SendValue {
  // Put inbox_tags in the send action instead of the receive action since all
  // we want to do is test the `ActionTesting::get_inbox_tag` function.
  using inbox_tags = tmpl::list<Tags::ValueTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::receive_data<Tags::ValueTag>(
        Parallel::get_parallel_component<ParallelComponent>(cache)[0], 1_st,
        23);
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Testing,
      tmpl::list<Actions::SendValue>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  enum class Phase { Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.ActionTesting.GetInboxTags", "[Unit]") {
  using metavars = Metavariables;
  using component = Component<metavars>;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  CHECK(ActionTesting::get_inbox_tag<component, Tags::ValueTag>(
            make_not_null(&runner), 0)
            .empty());
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(
      /// [const get inbox tags]
      ActionTesting::get_inbox_tag<component, Tags::ValueTag>(runner, 0).at(1)
      /// [const get inbox tags]
      == 23);
  /// [get inbox tags]
  ActionTesting::get_inbox_tag<component, Tags::ValueTag>(
      make_not_null(&runner), 0)
      .clear();
  /// [get inbox tags]
  CHECK(ActionTesting::get_inbox_tag<component, Tags::ValueTag>(
            make_not_null(&runner), 0)
            .empty());
}
}  // namespace InboxTags

namespace TestComponentMocking {
struct ValueTag : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "ValueTag"; }
};

template <typename Metavariables>
struct ComponentA {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using component_being_mocked = void;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
};

template <typename Metavariables>
struct ComponentB;

/// [mock component b]
template <typename Metavariables>
struct ComponentBMock {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using component_being_mocked = ComponentB<Metavariables>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<db::AddSimpleTags<ValueTag>>>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
};
/// [mock component b]

struct ActionCalledOnComponentB {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(
      db::DataBox<DbTagsList>& box,                          // NOLINT
      Parallel::GlobalCache<Metavariables>& /*cache*/,  // NOLINT
      const ArrayIndex& /*array_index*/) noexcept {
    db::mutate<ValueTag>(make_not_null(&box), [
    ](const gsl::not_null<int*> value) noexcept { *value = 5; });
  }
};

/// [action call component b]
struct CallActionOnComponentB {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& /*box*/,                  // NOLINT
                    Parallel::GlobalCache<Metavariables>& cache,  // NOLINT
                    const ArrayIndex& /*array_index*/) noexcept {
    Parallel::simple_action<ActionCalledOnComponentB>(
        Parallel::get_parallel_component<ComponentB<Metavariables>>(cache));
  }
};
/// [action call component b]

struct Metavariables {
  using component_list =
      tmpl::list<ComponentA<Metavariables>, ComponentBMock<Metavariables>>;

  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.ActionTesting.MockComponent", "[Unit]") {
  using metavars = Metavariables;
  using component_a = ComponentA<metavars>;
  using component_b_mock = ComponentBMock<metavars>;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component<component_a>(&runner, 0);
  /// [initialize component b]
  ActionTesting::emplace_component_and_initialize<
      ComponentBMock<Metavariables>>(&runner, 0, {0});
  /// [initialize component b]

  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);
  CHECK(ActionTesting::get_databox_tag<component_b_mock, ValueTag>(runner, 0) ==
        0);
  ActionTesting::simple_action<component_a, CallActionOnComponentB>(
      make_not_null(&runner), 0);
  /// [component b mock checks]
  CHECK(not ActionTesting::is_simple_action_queue_empty<component_b_mock>(
      runner, 0));
  ActionTesting::invoke_queued_simple_action<component_b_mock>(
      make_not_null(&runner), 0);
  CHECK(ActionTesting::get_databox_tag<component_b_mock, ValueTag>(runner, 0) ==
        5);
  /// [component b mock checks]
}
}  // namespace TestComponentMocking
}  // namespace
