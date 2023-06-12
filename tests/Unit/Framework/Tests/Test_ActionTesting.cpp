// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

// IWYU pragma: no_include <lrtslock.h>

namespace {
namespace TestSimpleAndThreadedActions {
struct MockSingleton {
  using chare_type = ActionTesting::MockSingletonChare;
};
struct MockArray {
  using chare_type = ActionTesting::MockArrayChare;
};
struct MockGroup {
  using chare_type = ActionTesting::MockGroupChare;
};
struct MockNodegroup {
  using chare_type = ActionTesting::MockNodeGroupChare;
};

static_assert(Parallel::is_singleton_v<MockSingleton>);
static_assert(Parallel::is_array_v<MockArray>);
static_assert(Parallel::is_group_v<MockGroup>);
static_assert(Parallel::is_nodegroup_v<MockNodegroup>);

struct simple_action_a;
struct simple_action_a_mock;
struct simple_action_c;
struct simple_action_c_mock;

struct threaded_action_b;
struct threaded_action_b_mock;

// [tags for const global cache]
struct ValueTag : db::SimpleTag {
  using type = int;
  static std::string name() { return "ValueTag"; }
};

struct PassedToB : db::SimpleTag {
  using type = double;
  static std::string name() { return "PassedToB"; }
};
// [tags for const global cache]

template <typename Metavariables>
struct component_for_simple_action_mock {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<ValueTag, PassedToB>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;

  // [simple action replace]
  using replace_these_simple_actions =
      tmpl::list<simple_action_a, simple_action_c, threaded_action_b>;
  using with_these_simple_actions =
      tmpl::list<simple_action_a_mock, simple_action_c_mock,
                 threaded_action_b_mock>;
  // [simple action replace]
  // [threaded action replace]
  using replace_these_threaded_actions = tmpl::list<threaded_action_b>;
  using with_these_threaded_actions = tmpl::list<threaded_action_b_mock>;
  // [threaded action replace]
};

struct simple_action_a_mock {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const int value) {
    db::mutate<ValueTag>(
        [&value](const gsl::not_null<int*> value_box) { *value_box = value; },
        make_not_null(&box));
  }
};

struct simple_action_b {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, PassedToB>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,                 // NOLINT
                    Parallel::GlobalCache<Metavariables>& cache,  // NOLINT
                    const ArrayIndex& /*array_index*/, const double to_call) {
    // simple_action_b is the action that we are testing, but it calls some
    // other actions that we don't want to test. Those are mocked where the body
    // of the mock actions records something in the DataBox that we check to
    // verify the mock actions were actually called.
    //
    // We do some "work" here by updating the `PassedToB` tag in the DataBox.
    db::mutate<PassedToB>(
        [&to_call](const gsl::not_null<double*> passed_to_b) {
          *passed_to_b = to_call;
        },
        make_not_null(&box));
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
                    const ArrayIndex& /*array_index*/) {
    db::mutate<ValueTag>(
        [](const gsl::not_null<int*> value_box) { *value_box = 25; },
        make_not_null(&box));
  }
};

struct threaded_action_a {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/) {
    db::mutate<ValueTag>(
        [](const gsl::not_null<int*> value_box) { *value_box = 35; },
        make_not_null(&box));
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
                    const int tag) {
    node_lock->lock();
    db::mutate<ValueTag>(
        [tag](const gsl::not_null<int*> value_box) { *value_box = tag; },
        make_not_null(&box));
    node_lock->unlock();
  }
};

struct SimpleActionMockMetavariables {
  using component_list = tmpl::list<
      component_for_simple_action_mock<SimpleActionMockMetavariables>>;

};

template <typename Metavariables>
struct component_for_global_cache_tags {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct MockMetavariablesWithGlobalCacheTags {
  using component_list = tmpl::list<
      component_for_global_cache_tags<MockMetavariablesWithGlobalCacheTags>>;
  // [const global cache metavars]
  using const_global_cache_tags = tmpl::list<ValueTag, PassedToB>;
  // [const global cache metavars]
};

void test_mock_runtime_system_constructors() {
  using metavars = MockMetavariablesWithGlobalCacheTags;
  using component = component_for_global_cache_tags<metavars>;
  // Test whether we can construct with tagged tuples in different orders.
  // [constructor const global cache tags known]
  ActionTesting::MockRuntimeSystem<metavars> runner1{{3, 7.0}};
  // [constructor const global cache tags known]
  // [constructor const global cache tags unknown]
  ActionTesting::MockRuntimeSystem<metavars> runner2{
      tuples::TaggedTuple<PassedToB, ValueTag>{7.0, 3}};
  // [constructor const global cache tags unknown]
  ActionTesting::emplace_component<component>(&runner1, 0);
  ActionTesting::emplace_component<component>(&runner2, 0);
  const auto& cache1 = ActionTesting::cache<component>(runner1, 0_st);
  const auto& cache2 = ActionTesting::cache<component>(runner2, 0_st);
  CHECK(Parallel::get<ValueTag>(cache1) == Parallel::get<ValueTag>(cache2));
  CHECK(Parallel::get<PassedToB>(cache1) == Parallel::get<PassedToB>(cache2));
  CHECK(Parallel::get<ValueTag>(cache1) == 3);
  CHECK(Parallel::get<PassedToB>(cache2) == 7);
}

SPECTRE_TEST_CASE("Unit.ActionTesting.MockSimpleAction", "[Unit]") {
  using metavars = SimpleActionMockMetavariables;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component_and_initialize<
      component_for_simple_action_mock<metavars>>(&runner, 0, {0, -1});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // [get databox]
  const auto& box =
      ActionTesting::get_databox<component_for_simple_action_mock<metavars>>(
          runner, 0);
  // [get databox]
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
  // [invoke simple action]
  ActionTesting::simple_action<component_for_simple_action_mock<metavars>,
                               simple_action_b>(make_not_null(&runner), 0, 2);
  // [invoke simple action]
  REQUIRE(not
          // [simple action queue empty]
          ActionTesting::is_simple_action_queue_empty<
              component_for_simple_action_mock<metavars>>(runner, 0)
          // [simple action queue empty]
  );
  // [invoke queued simple action]
  ActionTesting::invoke_queued_simple_action<
      component_for_simple_action_mock<metavars>>(make_not_null(&runner), 0);
  // [invoke queued simple action]
  CHECK(db::get<PassedToB>(box) == 2);
  CHECK(db::get<ValueTag>(box) == 25);
  REQUIRE(runner.is_simple_action_queue_empty<
          component_for_simple_action_mock<metavars>>(0));
  runner.queue_simple_action<component_for_simple_action_mock<metavars>,
                             simple_action_b>(0, 1);
  runner
      .invoke_queued_simple_action<component_for_simple_action_mock<metavars>>(
          0);
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

  // [invoke threaded action]
  ActionTesting::threaded_action<component_for_simple_action_mock<metavars>,
                                 threaded_action_a>(make_not_null(&runner), 0);
  // [invoke threaded action]
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
  static std::string name() { return "DummyTime"; }
  using type = int;
};

struct Action0 {
  using simple_tags = tmpl::list<DummyTimeTag>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 6);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<Action0>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

};

SPECTRE_TEST_CASE("Unit.ActionTesting.IsRetrievable", "[Unit]") {
  using metavars = Metavariables;
  using component = Component<Metavariables>;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  CHECK(ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner, 0));
  // Runs Action0
  runner.next_action<component>(0);
  CHECK(
      // [tag is retrievable]
      ActionTesting::tag_is_retrievable<component, DummyTimeTag>(runner, 0)
      // [tag is retrievable]
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
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Parallel::receive_data<Tags::ValueTag>(
        Parallel::get_parallel_component<ParallelComponent>(cache)[0], 1_st,
        23);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Testing,
                                        tmpl::list<Actions::SendValue>>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

};

SPECTRE_TEST_CASE("Unit.ActionTesting.GetInboxTags", "[Unit]") {
  using metavars = Metavariables;
  using component = Component<metavars>;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  CHECK(ActionTesting::get_inbox_tag<component, Tags::ValueTag>(
            make_not_null(&runner), 0)
            .empty());
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(
      // [const get inbox tags]
      ActionTesting::get_inbox_tag<component, Tags::ValueTag>(runner, 0).at(1)
      // [const get inbox tags]
      == 23);
  // [get inbox tags]
  ActionTesting::get_inbox_tag<component, Tags::ValueTag>(
      make_not_null(&runner), 0)
      .clear();
  // [get inbox tags]
  CHECK(ActionTesting::get_inbox_tag<component, Tags::ValueTag>(
            make_not_null(&runner), 0)
            .empty());
}
}  // namespace InboxTags

namespace TestComponentMocking {
struct ValueTag : db::SimpleTag {
  using type = int;
  static std::string name() { return "ValueTag"; }
};

template <typename Metavariables>
struct ComponentA {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

template <typename Metavariables>
struct ComponentB;

// [mock component b]
template <typename Metavariables>
struct ComponentBMock {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using component_being_mocked = ComponentB<Metavariables>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<ValueTag>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};
// [mock component b]

struct ActionCalledOnComponentB {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,                     // NOLINT
                    Parallel::GlobalCache<Metavariables>& /*cache*/,  // NOLINT
                    const ArrayIndex& /*array_index*/) {
    db::mutate<ValueTag>([](const gsl::not_null<int*> value) { *value = 5; },
                         make_not_null(&box));
  }
};

// [action call component b]
struct CallActionOnComponentB {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& /*box*/,             // NOLINT
                    Parallel::GlobalCache<Metavariables>& cache,  // NOLINT
                    const ArrayIndex& /*array_index*/) {
    Parallel::simple_action<ActionCalledOnComponentB>(
        Parallel::get_parallel_component<ComponentB<Metavariables>>(cache));
  }
};
// [action call component b]

struct Metavariables {
  using component_list =
      tmpl::list<ComponentA<Metavariables>, ComponentBMock<Metavariables>>;

};

SPECTRE_TEST_CASE("Unit.ActionTesting.MockComponent", "[Unit]") {
  using metavars = Metavariables;
  using component_a = ComponentA<metavars>;
  using component_b_mock = ComponentBMock<metavars>;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component<component_a>(&runner, 0);
  // [initialize component b]
  ActionTesting::emplace_component_and_initialize<
      ComponentBMock<Metavariables>>(&runner, 0, {0});
  // [initialize component b]

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  CHECK(ActionTesting::get_databox_tag<component_b_mock, ValueTag>(runner, 0) ==
        0);
  ActionTesting::queue_simple_action<component_a, CallActionOnComponentB>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_simple_action<component_a>(
      make_not_null(&runner), 0);
  // [component b mock checks]
  CHECK(not ActionTesting::is_simple_action_queue_empty<component_b_mock>(
      runner, 0));
  ActionTesting::invoke_queued_simple_action<component_b_mock>(
      make_not_null(&runner), 0);
  CHECK(ActionTesting::get_databox_tag<component_b_mock, ValueTag>(runner, 0) ==
        5);
  // [component b mock checks]
}
}  // namespace TestComponentMocking

namespace TestNodesAndCores {

struct ValueTag : db::SimpleTag {
  using type = int;
};

struct ValueTagSizeT : db::SimpleTag {
  using type = size_t;
};

template <typename Metavariables>
struct ComponentA {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<ValueTag, ValueTagSizeT>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct MyProc {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    return Parallel::my_proc<RetType>(*Parallel::local(my_proxy[array_index]));
  }
};

struct MyNode {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    return Parallel::my_node<RetType>(*Parallel::local(my_proxy[array_index]));
  }
};

struct LocalRank {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    return Parallel::my_local_rank<RetType>(
        *Parallel::local(my_proxy[array_index]));
  }
};

struct NumProcs {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    return Parallel::number_of_procs<RetType>(
        *Parallel::local(my_proxy[array_index]));
  }
};

struct NumNodes {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    return Parallel::number_of_nodes<RetType>(
        *Parallel::local(my_proxy[array_index]));
  }
};

template <int NodeIndex>
struct ProcsOnNode {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    const RetType value_from_int = Parallel::procs_on_node<RetType>(
        static_cast<int>(NodeIndex), *Parallel::local(my_proxy[array_index]));
    const RetType value_from_size_t = Parallel::procs_on_node<RetType>(
        static_cast<size_t>(NodeIndex),
        *Parallel::local(my_proxy[array_index]));
    CHECK(value_from_int == value_from_size_t);
    return value_from_int;
  }
};

template <int NodeIndex>
struct FirstProcOnNode {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    const RetType value_from_int = Parallel::first_proc_on_node<RetType>(
        static_cast<int>(NodeIndex), *Parallel::local(my_proxy[array_index]));
    const RetType value_from_size_t = Parallel::first_proc_on_node<RetType>(
        static_cast<size_t>(NodeIndex),
        *Parallel::local(my_proxy[array_index]));
    CHECK(value_from_int == value_from_size_t);
    return value_from_int;
  }
};

template <int ProcIndex>
struct NodeOf {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    const RetType value_from_int = Parallel::node_of<RetType>(
        static_cast<int>(ProcIndex), *Parallel::local(my_proxy[array_index]));
    const RetType value_from_size_t =
        Parallel::node_of<RetType>(static_cast<size_t>(ProcIndex),
                                   *Parallel::local(my_proxy[array_index]));
    CHECK(value_from_int == value_from_size_t);
    return value_from_int;
  }
};

template <int ProcIndex>
struct LocalRankOf {
  template <typename MyProxy, typename ArrayIndex, typename RetType>
  static auto f(MyProxy& my_proxy, const ArrayIndex& array_index) -> RetType {
    const RetType value_from_int = Parallel::local_rank_of<RetType>(
        static_cast<int>(ProcIndex), *Parallel::local(my_proxy[array_index]));
    const RetType value_from_size_t = Parallel::local_rank_of<RetType>(
        static_cast<size_t>(ProcIndex),
        *Parallel::local(my_proxy[array_index]));
    CHECK(value_from_int == value_from_size_t);
    return value_from_int;
  }
};

template <typename Tag, typename Functor>
struct ActionSetValueTo {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, Tag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    using ProxyType = std::decay_t<decltype(my_proxy)>;
    using T = typename Tag::type;
    // We must specify all templates here explicitly otherwise it won't build
    T function_value =
        Functor::template f<ProxyType, ArrayIndex, T>(my_proxy, array_index);
    db::mutate<Tag>(
        [&function_value](const gsl::not_null<T*> value) {
          *value = function_value;
        },
        make_not_null(&box));
  }
};

struct MetavariablesOneComponent {
  using component_list = tmpl::list<ComponentA<MetavariablesOneComponent>>;

};

void test_parallel_info_functions() {
  using metavars = MetavariablesOneComponent;
  using component_a = ComponentA<metavars>;

  // Choose 2 nodes with 3 cores on first node and 2 cores on second node.
  const int num_nodes = 2;
  const int procs_node_0 = 3;
  const int procs_node_1 = 2;
  const int num_procs = procs_node_0 + procs_node_1;
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {},
      {},
      {static_cast<size_t>(procs_node_0), static_cast<size_t>(procs_node_1)}};

  // Choose array indices by hand, not in simple order, and choose
  // arbitrary initial values.
  ActionTesting::emplace_array_component_and_initialize<component_a>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      {-1, 101_st});
  ActionTesting::emplace_array_component_and_initialize<component_a>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{1}, 2,
      {-2, 102_st});
  ActionTesting::emplace_array_component_and_initialize<component_a>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{2}, 4,
      {-3, 103_st});
  ActionTesting::emplace_array_component_and_initialize<component_a>(
      &runner, ActionTesting::NodeId{1}, ActionTesting::LocalCoreId{0}, 1,
      {-4, 104_st});
  ActionTesting::emplace_array_component_and_initialize<component_a>(
      &runner, ActionTesting::NodeId{1}, ActionTesting::LocalCoreId{1}, 3,
      {-5, 105_st});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Check that initial values are correct.
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTag>(runner, 0) == -1);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTag>(runner, 1) == -4);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTag>(runner, 2) == -2);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTag>(runner, 3) == -5);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTag>(runner, 4) == -3);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTagSizeT>(runner, 0) ==
        101);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTagSizeT>(runner, 1) ==
        104);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTagSizeT>(runner, 2) ==
        102);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTagSizeT>(runner, 3) ==
        105);
  CHECK(ActionTesting::get_databox_tag<component_a, ValueTagSizeT>(runner, 4) ==
        103);

  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<ValueTag, MyProc>>(
        make_not_null(&runner), i);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<ValueTagSizeT, MyProc>>(
        make_not_null(&runner), i);
  }
  using tag_list = tmpl::list<ValueTag, ValueTagSizeT>;
  // Should all be set to proc (which Mark computed by hand)
  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 0) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 1) == 3);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 2) == 1);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 3) == 4);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 4) == 2);
  });

  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<ValueTag, MyNode>>(
        make_not_null(&runner), i);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<ValueTagSizeT, MyNode>>(
        make_not_null(&runner), i);
  }
  // Should all be set to node (which Mark computed by hand)
  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 0) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 1) == 1);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 2) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 3) == 1);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 4) == 0);
  });

  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<ValueTag, LocalRank>>(
        make_not_null(&runner), i);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<ValueTagSizeT, LocalRank>>(
        make_not_null(&runner), i);
  }
  // Should all be set to local rank (which Mark computed by hand)
  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 0) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 1) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 2) == 1);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 3) == 1);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 4) == 2);
  });

  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    for (size_t i = 0; i < 5; ++i) {
      ActionTesting::simple_action<component_a,
                                   ActionSetValueTo<tag, NumProcs>>(
          make_not_null(&runner), i);
      CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, i) == 5);
    }
    for (size_t i = 0; i < 5; ++i) {
      ActionTesting::simple_action<component_a,
                                   ActionSetValueTo<tag, NumNodes>>(
          make_not_null(&runner), i);
      CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, i) == 2);
    }
  });

  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    // procs_on_node for the 2 nodes.
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, ProcsOnNode<0>>>(
        make_not_null(&runner), 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 0) == 3);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, ProcsOnNode<1>>>(
        make_not_null(&runner), 2);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 2) == 2);

    // first_proc_on_node for the 2 nodes.
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, FirstProcOnNode<0>>>(
        make_not_null(&runner), 1);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 1) == 0);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, FirstProcOnNode<1>>>(
        make_not_null(&runner), 3);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 3) == 3);
  });

  // node_of
  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    ActionTesting::simple_action<component_a, ActionSetValueTo<tag, NodeOf<0>>>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<component_a, ActionSetValueTo<tag, NodeOf<1>>>(
        make_not_null(&runner), 1);
    ActionTesting::simple_action<component_a, ActionSetValueTo<tag, NodeOf<2>>>(
        make_not_null(&runner), 2);
    ActionTesting::simple_action<component_a, ActionSetValueTo<tag, NodeOf<3>>>(
        make_not_null(&runner), 3);
    ActionTesting::simple_action<component_a, ActionSetValueTo<tag, NodeOf<4>>>(
        make_not_null(&runner), 4);
  });
  // Check if set to values that Mark computed by hand.
  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 0) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 1) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 2) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 3) == 1);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 4) == 1);
  });

  // local_rank_of
  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, LocalRankOf<0>>>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, LocalRankOf<1>>>(
        make_not_null(&runner), 1);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, LocalRankOf<2>>>(
        make_not_null(&runner), 2);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, LocalRankOf<3>>>(
        make_not_null(&runner), 3);
    ActionTesting::simple_action<component_a,
                                 ActionSetValueTo<tag, LocalRankOf<4>>>(
        make_not_null(&runner), 4);
  });
  // Check if set to values that Mark computed by hand.
  tmpl::for_each<tag_list>([&runner](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 0) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 1) == 1);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 2) == 2);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 3) == 0);
    CHECK(ActionTesting::get_databox_tag<component_a, tag>(runner, 4) == 1);
  });

  // Check the parallel info functions of the GlobalCache in the testing
  // framework
  auto& cache = ActionTesting::cache<component_a>(runner, 0_st);
  CHECK(cache.number_of_procs() == num_procs);
  CHECK(cache.number_of_nodes() == num_nodes);
  CHECK(cache.procs_on_node(0) == procs_node_0);
  CHECK(cache.procs_on_node(1) == procs_node_1);
  CHECK(cache.first_proc_on_node(0) == 0);
  CHECK(cache.first_proc_on_node(1) == procs_node_0);
  CHECK(Parallel::number_of_procs<int>(cache) == num_procs);
  CHECK(Parallel::number_of_nodes<int>(cache) == num_nodes);
  CHECK(Parallel::procs_on_node<int>(0, cache) == procs_node_0);
  CHECK(Parallel::procs_on_node<int>(1, cache) == procs_node_1);
  CHECK(Parallel::first_proc_on_node<int>(0, cache) == 0);
  CHECK(Parallel::first_proc_on_node<int>(1, cache) == procs_node_0);
  for (int i = 0; i < num_procs; i++) {
    CHECK(cache.node_of(i) == (i < procs_node_0 ? 0 : 1));
    CHECK(cache.local_rank_of(i) == (i < procs_node_0 ? i : i - procs_node_0));
    CHECK(Parallel::node_of<int>(i, cache) == (i < procs_node_0 ? 0 : 1));
    CHECK(Parallel::local_rank_of<int>(i, cache) ==
          (i < procs_node_0 ? i : i - procs_node_0));
  }
  for (int i = 0; i < num_procs; i++) {
    auto& local_cache =
        ActionTesting::cache<component_a>(runner, static_cast<size_t>(i));
    auto& proxy = Parallel::get_parallel_component<component_a>(local_cache);
    auto& local_obj = *Parallel::local(proxy[static_cast<size_t>(i)]);
    const int my_proc = Parallel::my_proc<int>(local_obj);
    CHECK(local_cache.my_proc() == my_proc);
    CHECK(local_cache.my_node() == (my_proc < procs_node_0 ? 0 : 1));
    CHECK(local_cache.my_local_rank() ==
          (my_proc < procs_node_0 ? my_proc : my_proc - procs_node_0));
    CHECK(Parallel::my_proc<int>(local_cache) == my_proc);
    CHECK(Parallel::my_node<int>(local_cache) ==
          (my_proc < procs_node_0 ? 0 : 1));
    CHECK(Parallel::my_local_rank<int>(local_cache) ==
          (my_proc < procs_node_0 ? my_proc : my_proc - procs_node_0));
  }
}

template <typename Metavariables>
struct GroupComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<ValueTag>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct MetavariablesGroupComponent {
  using component_list =
      tmpl::list<GroupComponent<MetavariablesGroupComponent>>;

};

void test_group_emplace() {
  using metavars = MetavariablesGroupComponent;
  using component = GroupComponent<metavars>;

  // Choose 2 nodes with 3 cores on first node and 2 cores on second node.
  ActionTesting::MockRuntimeSystem<metavars> runner{{}, {}, {3, 2}};

  ActionTesting::emplace_group_component_and_initialize<component>(&runner,
                                                                   {-3});

  // Check initial values for all components of the group.
  for (size_t i = 0; i < 5; ++i) {
    CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, i) == -3);
  }

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Number of procs should be 5 for all indices.
  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::simple_action<component,
                                 ActionSetValueTo<ValueTag, NumProcs>>(
        make_not_null(&runner), i);
    CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, i) == 5);
  }

  // Number of nodes should be 2 for all indices.
  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::simple_action<component,
                                 ActionSetValueTo<ValueTag, NumNodes>>(
        make_not_null(&runner), i);
    CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, i) == 2);
  }

  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::simple_action<component, ActionSetValueTo<ValueTag, MyProc>>(
        make_not_null(&runner), i);
  }

  // Should all be set to proc (which Mark computed by hand)
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 0) == 0);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 1) == 1);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 2) == 2);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 3) == 3);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 4) == 4);

  for (size_t i = 0; i < 5; ++i) {
    ActionTesting::simple_action<component, ActionSetValueTo<ValueTag, MyNode>>(
        make_not_null(&runner), i);
  }
  // Should all be set to node (which Mark computed by hand)
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 0) == 0);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 1) == 0);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 2) == 0);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 3) == 1);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 4) == 1);
}

template <typename Metavariables>
struct NodeGroupComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 db::AddSimpleTags<ValueTag>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
};

struct MetavariablesNodeGroupComponent {
  using component_list =
      tmpl::list<NodeGroupComponent<MetavariablesNodeGroupComponent>>;

};

void test_nodegroup_emplace() {
  using metavars = MetavariablesNodeGroupComponent;
  using component = NodeGroupComponent<metavars>;

  // Choose 2 nodes with 3 cores on first node and 2 cores on second node.
  ActionTesting::MockRuntimeSystem<metavars> runner{{}, {}, {3, 2}};

  ActionTesting::emplace_nodegroup_component_and_initialize<component>(&runner,
                                                                       {-3});

  // Check initial values for all components of the nodegroup.
  for (size_t i = 0; i < 2; ++i) {
    CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, i) == -3);
  }

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Number of procs should be 5 for all indices.
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::simple_action<component,
                                 ActionSetValueTo<ValueTag, NumProcs>>(
        make_not_null(&runner), i);
    CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, i) == 5);
  }

  // Number of nodes should be 2 for all indices.
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::simple_action<component,
                                 ActionSetValueTo<ValueTag, NumNodes>>(
        make_not_null(&runner), i);
    CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, i) == 2);
  }

  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::simple_action<component, ActionSetValueTo<ValueTag, MyProc>>(
        make_not_null(&runner), i);
  }

  // Should all be set to proc (which Mark computed by hand)
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 0) == 0);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 1) == 3);

  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::simple_action<component, ActionSetValueTo<ValueTag, MyNode>>(
        make_not_null(&runner), i);
  }
  // Should all be set to node (which Mark computed by hand)
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 0) == 0);
  CHECK(ActionTesting::get_databox_tag<component, ValueTag>(runner, 1) == 1);
}

struct MetavariablesWithPup {
  using component_list = tmpl::list<NodeGroupComponent<MetavariablesWithPup>>;


  void pup(PUP::er& /*p*/) {}
};

void test_sizing() {
  using metavars = MetavariablesWithPup;
  using component = NodeGroupComponent<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_nodegroup_component_and_initialize<component>(&runner,
                                                                       {-3});
  auto& cache = ActionTesting::cache<component>(runner, 0_st);
  auto& proxy = Parallel::get_parallel_component<component>(cache);
  auto& local_branch = *Parallel::local_branch(proxy);

  // The fact that this doesn't cause an error means it is successful. We aren't
  // concerned with the actual value
  const size_t size = size_of_object_in_bytes(local_branch);
  (void)size;

  CHECK_THROWS_WITH(
      ([&local_branch]() { serialize(local_branch); }()),
      Catch::Contains("MockDistributedObject is not serializable. This pup "
                      "member can only be used for sizing."));
}

SPECTRE_TEST_CASE("Unit.ActionTesting.NodesAndCores", "[Unit]") {
  test_parallel_info_functions();
  test_group_emplace();
  test_nodegroup_emplace();
  test_sizing();
}

}  // namespace TestNodesAndCores

namespace TestMutableGlobalCache {
// [mutable cache tag]
struct CacheTag : db::SimpleTag {
  using type = int;
};
// [mutable cache tag]

template <int Value>
struct CacheTagUpdater {
  static void apply(gsl::not_null<typename CacheTag::type*> tag_value) {
    *tag_value = Value;
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

// This Action does nothing other than set a bool so that we can
// check if it was called.
static bool simple_action_to_test_was_called = false;
struct SimpleActionToTest {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(const db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    simple_action_to_test_was_called = true;
  }
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  // [mutable global cache metavars]
  using mutable_global_cache_tags = tmpl::list<CacheTag>;
  // [mutable global cache metavars]

};

SPECTRE_TEST_CASE("Unit.ActionTesting.MutableGlobalCache", "[Unit]") {
  using metavars = Metavariables;
  using component = Component<metavars>;

  // [mutable global cache runner]
  ActionTesting::MockRuntimeSystem<metavars> runner{{}, {0}};
  // [mutable global cache runner]
  ActionTesting::emplace_array_component<component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0);

  auto& cache = ActionTesting::cache<component>(runner, 0);
  auto& element_proxy = ::Parallel::get_parallel_component<component>(cache)[0];

  CHECK(Parallel::mutable_cache_item_is_ready<CacheTag>(
      cache, [](const int /*value*/) {
        return std::unique_ptr<Parallel::Callback>{};
      }));

  CHECK(not Parallel::mutable_cache_item_is_ready<CacheTag>(
      cache, [&element_proxy](const int /*value*/) {
        return std::unique_ptr<Parallel::Callback>(
            new Parallel::PerformAlgorithmCallback<decltype(element_proxy)>(
                element_proxy));
      }));

  CHECK(not Parallel::mutable_cache_item_is_ready<CacheTag>(
      cache, [&element_proxy](const int /*value*/) {
        return std::unique_ptr<Parallel::Callback>(
            new Parallel::SimpleActionCallback<SimpleActionToTest,
                                               decltype(element_proxy)>(
                element_proxy));
      }));

  // Should be no queued simple actions.
  CHECK(ActionTesting::is_simple_action_queue_empty<component>(runner, 0));

  // After we mutate the item, then SimpleActionToTest should be queued...
  Parallel::mutate<CacheTag, CacheTagUpdater<1>>(cache);
  CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner, 0) ==
        1);
  // ... so invoke it
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);
  // ... and test that the action was called.
  CHECK(simple_action_to_test_was_called);

  // Now reset for another call.
  simple_action_to_test_was_called = false;

  // Simple actions can be called on entire components
  // (i.e. all elements at once as opposed to one element at a time).
  auto& all_elements_proxy =
      ::Parallel::get_parallel_component<component>(cache);
  CHECK(not Parallel::mutable_cache_item_is_ready<CacheTag>(
      cache, [&all_elements_proxy](const int /*value*/) {
        return std::unique_ptr<Parallel::Callback>(
            new Parallel::SimpleActionCallback<SimpleActionToTest,
                                               decltype(all_elements_proxy)>(
                all_elements_proxy));
      }));

  // Should be no queued simple actions.
  CHECK(ActionTesting::is_simple_action_queue_empty<component>(runner, 0));

  // After we mutate the item, then SimpleActionToTest should be queued...
  Parallel::mutate<CacheTag, CacheTagUpdater<3>>(cache);
  CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner, 0) ==
        1);
  // ... so invoke it
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);
  // ... and test that the action was called.
  CHECK(simple_action_to_test_was_called);

  // Currently perform_algorithm cannot be called on all elements at
  // once (in the ActionTesting framework), but this is not difficult
  // to add in the future.
}
}  // namespace TestMutableGlobalCache

}  // namespace
