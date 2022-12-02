// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

namespace {
struct TestMetavariables;
template <class Metavariables>
struct NoOpsComponent;

struct TestAlgorithmArrayInstance {
  explicit TestAlgorithmArrayInstance(int ii) : i(ii) {}
  TestAlgorithmArrayInstance() = default;
  int i = 0;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | i; }
};

bool operator==(const TestAlgorithmArrayInstance& lhs,
                const TestAlgorithmArrayInstance& rhs) {
  return lhs.i == rhs.i;
}

TestAlgorithmArrayInstance& operator++(TestAlgorithmArrayInstance& instance) {
  instance.i++;
  return instance;
}
}  // namespace

namespace std {
template <>
struct hash<TestAlgorithmArrayInstance> {
  size_t operator()(const TestAlgorithmArrayInstance& t) const {
    return hash<int>{}(t.i);
  }
};
}  // namespace std

namespace {
struct ElementIndex {};

struct CountActionsCalled : db::SimpleTag {
  static std::string name() { return "CountActionsCalled"; }
  using type = int;
};

struct Int0 : db::SimpleTag {
  static std::string name() { return "Int0"; }
  using type = int;
};

struct Int1 : db::SimpleTag {
  static std::string name() { return "Int1"; }
  using type = int;
};

struct TemporalId : db::SimpleTag {
  static std::string name() { return "TemporalId"; }
  using type = TestAlgorithmArrayInstance;
};

//////////////////////////////////////////////////////////////////////
// Test actions that do not add or remove from the DataBox
//////////////////////////////////////////////////////////////////////

namespace no_op_test {
struct increment_count_actions_called {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        std::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    static int a = 0;
    return {(++a >= 5 ? Parallel::AlgorithmExecution::Pause
                      : Parallel::AlgorithmExecution::Continue),
            std::nullopt};
  }
};

struct no_op {
  // [apply_iterative]
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/)
  // [apply_iterative]
  {
    static_assert(
        std::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct initialize {
  using simple_tags = tmpl::list<CountActionsCalled, Int0, Int1>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        std::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 0, 1, 100);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct finalize {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<
                std::is_same_v<CountActionsCalled, DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    static_assert(
        std::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 5);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == 1);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int1>(box) == 100);
  }
};
}  // namespace no_op_test

template <class Metavariables>
struct NoOpsComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        tmpl::list<no_op_test::initialize>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Register,
                     tmpl::list<no_op_test::increment_count_actions_called,
                                no_op_test::no_op>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    // [start_phase]
    Parallel::get_parallel_component<NoOpsComponent>(local_cache)
        .start_phase(next_phase);
    // [start_phase]
    if (next_phase == Parallel::Phase::Testing) {
      Parallel::simple_action<no_op_test::finalize>(
          Parallel::get_parallel_component<NoOpsComponent>(local_cache));
    }
  }
};

//////////////////////////////////////////////////////////////////////
// Adding and remove elements from DataBox tests
//////////////////////////////////////////////////////////////////////

namespace add_remove_test {
struct add_int_value_10 {
  using simple_tags = tmpl::list<Int0>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    static int a = 0;
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 10);
    return {(++a >= 5 ? Parallel::AlgorithmExecution::Pause
                      : Parallel::AlgorithmExecution::Continue),
            std::nullopt};
  }
};

struct increment_int0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    db::mutate<Int0>(make_not_null(&box),
                     [](const gsl::not_null<int*> int0) { ++*int0; });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct remove_int0 {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == 11);
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called, const int& int0) {
          SPECTRE_PARALLEL_REQUIRE(int0 == 11);
          ++*count_actions_called;
        },
        db::get<Int0>(box));
    // default assign to "remove"
    Initialization::mutate_assign<tmpl::list<Int0>>(make_not_null(&box), 0);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct test_args {
  // [requires_action]
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<CountActionsCalled,
                                              db::DataBox<DbTags>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const double v0,
                    std::vector<double>&& v1) {
    // [requires_action]
    SPECTRE_PARALLEL_REQUIRE(v0 == 4.82937);
    SPECTRE_PARALLEL_REQUIRE(v1 == (std::vector<double>{3.2, -8.4, 7.5}));
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 13);
  }
};

struct initialize {
  using simple_tags = tmpl::list<CountActionsCalled, TemporalId>;
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTagsList, CountActionsCalled>> =
          nullptr>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 0,
                                               TestAlgorithmArrayInstance{0});
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, CountActionsCalled>> = nullptr>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct finalize {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<
                std::is_same_v<CountActionsCalled, DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 13);
  }
};
}  // namespace add_remove_test

template <class Metavariables>
struct MutateComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using array_index = ElementIndex;  // Just to test nothing breaks
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<add_remove_test::initialize>>,
      Parallel::PhaseActions<Parallel::Phase::Solve,
                             tmpl::list<add_remove_test::add_int_value_10,
                                        add_remove_test::increment_int0,
                                        add_remove_test::remove_int0>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<MutateComponent>(local_cache)
        .start_phase(next_phase);
    if (next_phase == Parallel::Phase::Evolve) {
      // [simple_action_call]
      Parallel::simple_action<add_remove_test::test_args>(
          Parallel::get_parallel_component<MutateComponent>(local_cache),
          4.82937, std::vector<double>{3.2, -8.4, 7.5});
      // [simple_action_call]
      Parallel::simple_action<add_remove_test::finalize>(
          Parallel::get_parallel_component<MutateComponent>(local_cache));
    }
  }
};

//////////////////////////////////////////////////////////////////////
// Test receiving data
//////////////////////////////////////////////////////////////////////

namespace receive_data_test {
// [int receive tag insert]
struct IntReceiveTag
    : public Parallel::InboxInserters::MemberInsert<IntReceiveTag> {
  using temporal_id = TestAlgorithmArrayInstance;
  using type = std::unordered_map<temporal_id, std::unordered_multiset<int>>;
};
// [int receive tag insert]

struct add_int0_to_box {
  using simple_tags = tmpl::list<Int0>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Initialization::mutate_assign<tmpl::list<Int0>>(make_not_null(&box), 0);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct set_int0_from_receive {
  using inbox_tags = tmpl::list<IntReceiveTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<IntReceiveTag>(inboxes);
    db::mutate<Int1>(make_not_null(&box),
                     [](const gsl::not_null<int*> int1) { ++*int1; });
    // [retry_example]
    if (inbox.count(db::get<TemporalId>(box)) == 0) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    // [retry_example]

    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    static int a = 0;
    auto int0 = *inbox[db::get<TemporalId>(box)].begin();
    inbox.erase(db::get<TemporalId>(box));
    db::mutate<Int0>(
        make_not_null(&box),
        [&int0](const gsl::not_null<int*> int0_box) {
          *int0_box = int0;
        });
    return {++a >= 5 ? Parallel::AlgorithmExecution::Pause
                     : Parallel::AlgorithmExecution::Continue,
            std::nullopt};
  }
};

struct update_instance {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<TemporalId>(
        make_not_null(&box),
        [](const gsl::not_null<TestAlgorithmArrayInstance*> temporal_id) {
          ++*temporal_id;
        });
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct initialize {
  using simple_tags = tmpl::list<CountActionsCalled, Int1, TemporalId>;
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTagsList, CountActionsCalled>> =
          nullptr>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 0, 0,
                                               TestAlgorithmArrayInstance{0});
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, CountActionsCalled>> = nullptr>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct finalize {
  using inbox_tags = tmpl::list<IntReceiveTag>;
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<CountActionsCalled,
                                              db::DataBox<DbTags>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    SPECTRE_PARALLEL_REQUIRE(db::get<TemporalId>(box) ==
                             TestAlgorithmArrayInstance{4});
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 13);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int1>(box) == 10);
  }
};
}  // namespace receive_data_test

template <class Metavariables>
struct ReceiveComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using array_index = ElementIndex;  // Just to test nothing breaks
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<receive_data_test::initialize>>,
      Parallel::PhaseActions<
          Parallel::Phase::ImportInitialData,
          tmpl::list<receive_data_test::add_int0_to_box,
                     receive_data_test::set_int0_from_receive,
                     add_remove_test::increment_int0,
                     add_remove_test::remove_int0,
                     receive_data_test::update_instance>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<ReceiveComponent>(local_cache)
        .start_phase(next_phase);
    if (next_phase == Parallel::Phase::ImportInitialData) {
      for (TestAlgorithmArrayInstance instance{0};
           not(instance == TestAlgorithmArrayInstance{5}); ++instance) {
        int dummy_int = 10;
        Parallel::receive_data<receive_data_test::IntReceiveTag>(
            Parallel::get_parallel_component<ReceiveComponent>(local_cache),
            instance, dummy_int);
      }
    } else if (next_phase ==
               Parallel::Phase::InitializeInitialDataDependentQuantities) {
      Parallel::simple_action<receive_data_test::finalize>(
          Parallel::get_parallel_component<ReceiveComponent>(local_cache));
    }
  }
};

//////////////////////////////////////////////////////////////////////
// Test out of order execution of Actions
//////////////////////////////////////////////////////////////////////

template <class Metavariables>
struct AnyOrderComponent;

namespace any_order {
struct iterate_increment_int0 {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        std::is_same_v<ParallelComponent, AnyOrderComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    SPECTRE_PARALLEL_REQUIRE((db::get<CountActionsCalled>(box) - 1) / 2 ==
                             db::get<Int0>(box) - 10);

    const int max_int0_value = 25;
    if (db::get<Int0>(box) < max_int0_value) {
      return {
          Parallel::AlgorithmExecution::Continue,
          tmpl::index_of<ActionList, ::add_remove_test::increment_int0>::value};
    }

    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == max_int0_value);
    // [out_of_order_action]
    return {Parallel::AlgorithmExecution::Pause,
            tmpl::index_of<ActionList, iterate_increment_int0>::value + 1};
    // [out_of_order_action]
  }
};

struct finalize {
  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<
          tmpl2::flat_any_v<std::is_same_v<CountActionsCalled, DbTags>...> and
          tmpl2::flat_any_v<std::is_same_v<Int0, DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const Parallel::GlobalCache<Metavariables>&
                    /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    static_assert(
        std::is_same_v<ParallelComponent, AnyOrderComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<TemporalId>(box) ==
                             TestAlgorithmArrayInstance{0});
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 31);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == 25);
  }
};
}  // namespace any_order

template <class Metavariables>
struct AnyOrderComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using array_index = ElementIndex;  // Just to test nothing breaks

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<add_remove_test::initialize>>,
      Parallel::PhaseActions<Parallel::Phase::Execute,
                             tmpl::list<add_remove_test::add_int_value_10,
                                        add_remove_test::increment_int0,
                                        any_order::iterate_increment_int0,
                                        add_remove_test::remove_int0,
                                        receive_data_test::update_instance>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<AnyOrderComponent>(local_cache)
        .start_phase(next_phase);
    if (next_phase == Parallel::Phase::Cleanup) {
      Parallel::simple_action<any_order::finalize>(
          Parallel::get_parallel_component<AnyOrderComponent>(local_cache));
    }
  }
};

struct TestMetavariables {
  // [component_list_example]
  using component_list = tmpl::list<NoOpsComponent<TestMetavariables>,
                                    MutateComponent<TestMetavariables>,
                                    ReceiveComponent<TestMetavariables>,
                                    AnyOrderComponent<TestMetavariables>>;
  // [component_list_example]

  // [help_string_example]
  static constexpr Options::String help =
      "An executable for testing the core functionality of the Algorithm. "
      "Actions that do not perform any operations (no-ops), invoking simple "
      "actions, mutating data in the DataBox, receiving data from other "
      "parallel components, and out-of-order execution of Actions are all "
      "tested. All tests are run just by running the executable, no input file "
      "or command line arguments are required";
  // [help_string_example]

  static constexpr std::array<Parallel::Phase, 10> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::Testing, Parallel::Phase::Solve,
       Parallel::Phase::Evolve, Parallel::Phase::ImportInitialData,
       Parallel::Phase::InitializeInitialDataDependentQuantities,
       Parallel::Phase::Execute, Parallel::Phase::Cleanup,
       Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
}  // namespace

// [charm_init_funcs_example]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
// [charm_init_funcs_example]

// [charm_main_example]
using charmxx_main_component = Parallel::Main<TestMetavariables>;
// [charm_main_example]

// [[OutputRegex, DistributedObject has been constructed with a nullptr]]
SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.NullptrConstructError",
                  "[Parallel][Unit]") {
  ERROR_TEST();
  Parallel::DistributedObject<NoOpsComponent<TestMetavariables>,
                              tmpl::list<Parallel::PhaseActions<
                                  Parallel::Phase::Initialization,
                                  tmpl::list<add_remove_test::initialize>>>>{
      nullptr};
}

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
