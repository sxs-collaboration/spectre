// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
struct TestMetavariables;
template <class Metavariables>
struct NoOpsComponent;

struct TestAlgorithmArrayInstance {
  explicit TestAlgorithmArrayInstance(int ii) : i(ii) {}
  TestAlgorithmArrayInstance() = default;
  int i = 0;
  // clang-tidy: no non-const references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | i;
  }
};

bool operator==(const TestAlgorithmArrayInstance& lhs,
                const TestAlgorithmArrayInstance& rhs) noexcept {
  return lhs.i == rhs.i;
}

TestAlgorithmArrayInstance& operator++(
    TestAlgorithmArrayInstance& instance) noexcept {
  instance.i++;
  return instance;
}

namespace std {
template <>
struct hash<TestAlgorithmArrayInstance> {
  size_t operator()(const TestAlgorithmArrayInstance& t) const {
    return hash<int>{}(t.i);
  }
};
}  // namespace std

struct ElementId {};

struct CountActionsCalled : db::SimpleTag {
  static std::string name() noexcept { return "CountActionsCalled"; }
  using type = int;
};

struct Int0 : db::SimpleTag {
  static std::string name() noexcept { return "Int0"; }
  using type = int;
};

struct Int1 : db::SimpleTag {
  static std::string name() noexcept { return "Int1"; }
  using type = int;
};

struct TemporalId : db::SimpleTag {
  static std::string name() noexcept { return "TemporalId"; }
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
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        std::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    static int a = 0;
    return std::tuple<db::DataBox<DbTags>&&, bool>(std::move(box), ++a >= 5);
  }
};

struct no_op {
  // [apply_iterative]
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept
  // [apply_iterative]
  {
    static_assert(
        std::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::forward_as_tuple(std::move(box));
  }
};

struct initialize {
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTagsList, CountActionsCalled>> =
          nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        std::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<CountActionsCalled, Int0, Int1>>(
            std::move(box), 0, 1, 100),
        true);
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, CountActionsCalled>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box), true};
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
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<no_op_test::initialize>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::NoOpsStart,
          tmpl::list<no_op_test::increment_count_actions_called,
                     no_op_test::no_op>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    // [start_phase]
    Parallel::get_parallel_component<NoOpsComponent>(local_cache)
        .start_phase(next_phase);
    // [start_phase]
    if (next_phase == Metavariables::Phase::NoOpsFinish) {
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
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    static int a = 0;
    return std::make_tuple(
        db::create_from<tmpl::list<>, tmpl::list<Int0>>(std::move(box), 10),
        ++a >= 5);
  }
};

struct increment_int0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    db::mutate<Int0>(make_not_null(&box),
                     [](const gsl::not_null<int*> int0) { ++*int0; });
    return std::forward_as_tuple(std::move(box));
  }
};

struct remove_int0 {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == 11);
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called, const int& int0) {
          SPECTRE_PARALLEL_REQUIRE(int0 == 11);
          ++*count_actions_called;
        },
        db::get<Int0>(box));
    return std::make_tuple(db::create_from<tmpl::list<Int0>>(std::move(box)));
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
                    std::vector<double>&& v1) noexcept {
    // [requires_action]
    SPECTRE_PARALLEL_REQUIRE(v0 == 4.82937);
    SPECTRE_PARALLEL_REQUIRE(v1 == (std::vector<double>{3.2, -8.4, 7.5}));
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 13);
  }
};

struct initialize {
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTagsList, CountActionsCalled>> =
          nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<CountActionsCalled, TemporalId>>(
            std::move(box), 0, TestAlgorithmArrayInstance{0}),
        true);
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, CountActionsCalled>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box), true};
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
  using array_index = ElementId;  // Just to test nothing breaks
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<add_remove_test::initialize>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::MutateStart,
                             tmpl::list<add_remove_test::add_int_value_10,
                                        add_remove_test::increment_int0,
                                        add_remove_test::remove_int0>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;


  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<MutateComponent>(local_cache)
        .start_phase(next_phase);
    if (next_phase == Metavariables::Phase::MutateFinish) {
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

struct add_int0_from_receive {
  using inbox_tags = tmpl::list<IntReceiveTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<CountActionsCalled>(
        make_not_null(&box),
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        });
    static int a = 0;
    auto int0 = *(
        std::move(tuples::get<IntReceiveTag>(inboxes)[db::get<TemporalId>(box)])
            .begin());
    tuples::get<IntReceiveTag>(inboxes).erase(db::get<TemporalId>(box));
    return std::make_tuple(
        db::create_from<tmpl::list<>, tmpl::list<Int0>>(std::move(box), int0),
        ++a >= 5);
  }

  // [is_ready_example]
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept
  // [is_ready_example]
  {
    const auto& inbox = tuples::get<IntReceiveTag>(inboxes);
    // The const_cast in this function is purely for testing purposes, this is
    // NOT an example of how to use this function.
    // clang-tidy: do not use const_cast
    db::mutate<Int1>(
        make_not_null(&const_cast<db::DataBox<DbTags>&>(box)),  // NOLINT
        [](const gsl::not_null<int*> int1) { ++*int1; });
    return inbox.count(db::get<TemporalId>(box)) != 0;
  }
};

struct update_instance {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<TemporalId>(
        make_not_null(&box),
        [](const gsl::not_null<TestAlgorithmArrayInstance*> temporal_id) {
          ++*temporal_id;
        });
    return std::forward_as_tuple(std::move(box));
  }
};

struct initialize {
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTagsList, CountActionsCalled>> =
          nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        db::create_from<db::RemoveTags<>,
                        tmpl::list<CountActionsCalled, Int1, TemporalId>>(
            std::move(box), 0, 0, TestAlgorithmArrayInstance{0}),
        true);
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, CountActionsCalled>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box), true};
  }
};

struct finalize {
  using inbox_tags = tmpl::list<IntReceiveTag>;
  template <typename ParallelComponent, typename DbTags,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<CountActionsCalled,
                                              db::DataBox<DbTags>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) noexcept {
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
  using array_index = ElementId;  // Just to test nothing breaks
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<receive_data_test::initialize>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::ReceiveStart,
          tmpl::list<receive_data_test::add_int0_from_receive,
                     add_remove_test::increment_int0,
                     add_remove_test::remove_int0,
                     receive_data_test::update_instance>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ReceiveComponent>(local_cache)
        .start_phase(next_phase);
    if (next_phase == Metavariables::Phase::ReceiveStart) {
      for (TestAlgorithmArrayInstance instance{0};
           not(instance == TestAlgorithmArrayInstance{5}); ++instance) {
        int dummy_int = 10;
        Parallel::receive_data<receive_data_test::IntReceiveTag>(
            Parallel::get_parallel_component<ReceiveComponent>(local_cache),
            instance, dummy_int);
      }
    } else if (next_phase == Metavariables::Phase::ReceiveFinish) {
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
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>&
                    /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const  // NOLINT const
                    /*meta*/) noexcept
      -> std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool, size_t> {
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
      return std::tuple<decltype(std::move(box)), bool, size_t>(
          std::move(box), false,
          tmpl::index_of<ActionList, ::add_remove_test::increment_int0>::value);
    }

    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == max_int0_value);
    // [out_of_order_action]
    return std::tuple<decltype(std::move(box)), bool, size_t>(
        std::move(box), true,
        tmpl::index_of<ActionList, iterate_increment_int0>::value + 1);
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
  using array_index = ElementId;  // Just to test nothing breaks

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<add_remove_test::initialize>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::AnyOrderStart,
                             tmpl::list<add_remove_test::add_int_value_10,
                                        add_remove_test::increment_int0,
                                        any_order::iterate_increment_int0,
                                        add_remove_test::remove_int0,
                                        receive_data_test::update_instance>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<AnyOrderComponent>(local_cache)
        .start_phase(next_phase);
    if (next_phase == Metavariables::Phase::AnyOrderFinish) {
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
      "actions, mutating data in the DataBox, adding and removing items from "
      "the DataBox, receiving data from other parallel components, and "
      "out-of-order execution of Actions are all tested. All tests are run "
      "just by running the executable, no input file or command line arguments "
      "are required";
  // [help_string_example]

  // [determine_next_phase_example]
  enum class Phase {
    Initialization,
    NoOpsStart,
    NoOpsFinish,
    MutateStart,
    MutateFinish,
    ReceiveStart,
    ReceiveFinish,
    AnyOrderStart,
    AnyOrderFinish,
    Exit
  };

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<
          tuples::TaggedTuple<Tags...>*> /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::NoOpsStart;
      case Phase::NoOpsStart:
        return Phase::NoOpsFinish;
      case Phase::NoOpsFinish:
        return Phase::MutateStart;
      case Phase::MutateStart:
        return Phase::MutateFinish;
      case Phase::MutateFinish:
        return Phase::ReceiveStart;
      case Phase::ReceiveStart:
        return Phase::ReceiveFinish;
      case Phase::ReceiveFinish:
        return Phase::AnyOrderStart;
      case Phase::AnyOrderStart:
        return Phase::AnyOrderFinish;
      case Phase::AnyOrderFinish:
        [[fallthrough]];
      case Phase::Exit:
        return Phase::Exit;
      default:
        ERROR("Unknown Phase...");
    }

    return Phase::Exit;
  }
  // [determine_next_phase_example]
};

// [charm_init_funcs_example]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
// [charm_init_funcs_example]

// [charm_main_example]
using charmxx_main_component = Parallel::Main<TestMetavariables>;
// [charm_main_example]

// [[OutputRegex, AlgorithmImpl has been constructed with a nullptr]]
SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.NullptrConstructError",
                  "[Parallel][Unit]") {
  ERROR_TEST();
  Parallel::AlgorithmImpl < NoOpsComponent<TestMetavariables>,
      tmpl::list<
          Parallel::PhaseActions<typename TestMetavariables::Phase,
                                 TestMetavariables::Phase::Initialization,
                                 tmpl::list<add_remove_test::initialize>>>>{
          nullptr};
}

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
