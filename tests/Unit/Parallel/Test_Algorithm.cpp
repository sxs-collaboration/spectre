// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <unordered_map>

#include "AlgorithmSingleton.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Printf.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct TestMetavariables {
  using component_list = tmpl::list<>;
  enum class Phase { Initialization, Exit };
};

template <class Metavariables>
struct SingletonParallelComponent;

struct TestAlgorithmArrayInstance {
  explicit TestAlgorithmArrayInstance(int ii) : i(ii) {}
  TestAlgorithmArrayInstance() = default;
  int i = 0;
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
/// \cond
struct ElementId {};
/// \endcond

struct CountActionsCalled : db::DataBoxTag {
  static constexpr db::DataBoxString label = "CountActionsCalled";
  using type = int;
};

struct Int0 : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Int0";
  using type = int;
};

struct Int1 : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Int1";
  using type = int;
};

struct TemporalId : db::DataBoxTag {
  static constexpr db::DataBoxString label = "TemporalId";
  using type = TestAlgorithmArrayInstance;
};
}  // namespace

namespace {
namespace no_op_test {
struct increment_count_actions_called {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    static int a = 0;
    return std::tuple<db::DataBox<DbTags>&&, bool>(std::move(box), ++a >= 5);
  }
};

struct no_op {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::forward_as_tuple(std::move(box));
  }
};

struct initialize {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(
        db::create<typelist<CountActionsCalled, Int0, Int1>>(0, 1, 100));
  }
};

struct finalize {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<
                cpp17::is_same_v<CountActionsCalled, DbTags>...>> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    CHECK(db::get<CountActionsCalled>(box) == 5);
    CHECK(db::get<Int0>(box) == 1);
    CHECK(db::get<Int1>(box) == 100);
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace no_op_test
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.local_no_ops", "[Unit][Parallel]") {
  // Test 2 Actions that do not add or remove elements from a DataBox with 2
  // int tags
  Parallel::AlgorithmImpl<
      SingletonParallelComponent<TestMetavariables>,
      Parallel::Algorithms::Singleton, TestMetavariables,
      typelist<no_op_test::increment_count_actions_called, no_op_test::no_op>,
      ElementId,
      db::DataBox<
          db::get_databox_list<typelist<CountActionsCalled, Int0, Int1>>>>
      al_gore{};
  al_gore.template explicit_single_action<no_op_test::initialize>();
  al_gore.perform_algorithm();
  al_gore.template explicit_single_action<no_op_test::finalize>();
}

namespace {
namespace add_remove_test {
struct add_int_value_10 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    static int a = 0;
    return std::make_tuple(
        db::create_from<typelist<>, typelist<Int0>>(std::move(box), 10),
        ++a >= 5);
  }
};

struct increment_int0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    db::mutate<Int0>(box, [](int& int0) { int0++; });
    return std::forward_as_tuple(std::move(box));
  }
};

struct remove_int0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    CHECK(db::get<Int0>(box) == 11);
    db::mutate<CountActionsCalled>(
        box,
        [](int& count_actions_called, const int& int0) {
          CHECK(int0 == 11);
          count_actions_called++;
        },
        db::get<Int0>(box));
    return std::make_tuple(db::create_from<typelist<Int0>>(std::move(box)));
  }
};

struct test_args {
  using apply_args = tmpl::list<double, std::vector<double>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/, const double& v0,
                    std::vector<double>&& v1) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    CHECK(v0 == 4.82937);
    CHECK(v1 == (std::vector<double>{3.2, -8.4, 7.5}));
    return std::forward_as_tuple(box);
  }
};

struct initialize {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(db::create<typelist<CountActionsCalled, TemporalId>>(
        0, TestAlgorithmArrayInstance{0}));
  }
};

struct finalize {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<
                cpp17::is_same_v<CountActionsCalled, DbTags>...>> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    CHECK(db::get<CountActionsCalled>(box) == 13);
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace add_remove_test
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.local_mutate", "[Unit][Parallel]") {
  // Test 3 Actions where the first adds an item to the DataBox, the second
  // mutates it, and the third removes it.
  Parallel::AlgorithmImpl<
      SingletonParallelComponent<TestMetavariables>,
      Parallel::Algorithms::Singleton, TestMetavariables,
      typelist<add_remove_test::add_int_value_10,
               add_remove_test::increment_int0, add_remove_test::remove_int0>,
      ElementId,
      db::DataBox<
          db::get_databox_list<typelist<CountActionsCalled, TemporalId>>>>
      al_gore{};
  al_gore.template explicit_single_action<add_remove_test::initialize>();
  al_gore.perform_algorithm();
  al_gore.template explicit_single_action<add_remove_test::test_args>(
      std::make_tuple(4.82937, std::vector<double>{3.2, -8.4, 7.5}));
  al_gore.template explicit_single_action<add_remove_test::finalize>();
}

namespace {
namespace receive_data_test {
struct IntReceiveTag {
  using temporal_id = TestAlgorithmArrayInstance;
  using type = std::unordered_map<temporal_id, std::unordered_multiset<int>>;
};

struct add_int0_from_receive {
  using inbox_tags = tmpl::list<IntReceiveTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    static int a = 0;
    auto int0 = *(
        std::move(tuples::get<IntReceiveTag>(inboxes)[db::get<TemporalId>(box)])
            .begin());
    tuples::get<IntReceiveTag>(inboxes).erase(db::get<TemporalId>(box));
    return std::make_tuple(
        db::create_from<typelist<>, typelist<Int0>>(box, int0), ++a >= 5);
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox = tuples::get<IntReceiveTag>(inboxes);
    // The const_cast in this function is purely for testing purposes, this is
    // NOT an example of how to use this function. clang-tidy: do not use
    // const_cast
    db::mutate<Int1>(const_cast<db::DataBox<DbTags>&>(box),  // NOLINT
                     [](int& int1) { int1++; });
    return inbox.count(db::get<TemporalId>(box)) != 0;
  }
};

struct update_instance {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<TemporalId>(
        box, [](TestAlgorithmArrayInstance& temporal_id) { ++temporal_id; });
    return std::forward_as_tuple(std::move(box));
  }
};

struct initialize {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(
        db::create<typelist<CountActionsCalled, Int1, TemporalId>>(
            0, 0, TestAlgorithmArrayInstance{0}));
  }
};

struct finalize {
  using inbox_tags = tmpl::list<IntReceiveTag>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<
                cpp17::is_same_v<CountActionsCalled, DbTags>...>> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    CHECK(tuples::get<IntReceiveTag>(inboxes).empty());
    CHECK(db::get<TemporalId>(box) == TestAlgorithmArrayInstance{4});
    CHECK(db::get<CountActionsCalled>(box) == 13);
    CHECK(db::get<Int1>(box) == 10);
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace receive_data_test
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.local_receive", "[Unit][Parallel]") {
  // Test that actions block on receive correctly
  Parallel::AlgorithmImpl<
      SingletonParallelComponent<TestMetavariables>,
      Parallel::Algorithms::Singleton, TestMetavariables,
      typelist<receive_data_test::add_int0_from_receive,
               add_remove_test::increment_int0, add_remove_test::remove_int0,
               receive_data_test::update_instance>,
      ElementId,
      db::DataBox<
          db::get_databox_list<typelist<CountActionsCalled, Int1, TemporalId>>>>
      al_gore{};
  al_gore.template explicit_single_action<receive_data_test::initialize>();
  al_gore.perform_algorithm();
  for (TestAlgorithmArrayInstance instance{0};
       not(instance == TestAlgorithmArrayInstance{5}); ++instance) {
    int dummy_int = 10;
    al_gore.receive_data<receive_data_test::IntReceiveTag>(instance, dummy_int);
  }
  al_gore.template explicit_single_action<receive_data_test::finalize>();
}

namespace {
namespace any_order {
struct iterate_increment_int0 {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept
      -> std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool, size_t> {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    CHECK((db::get<CountActionsCalled>(box) - 1) / 2 ==
          db::get<Int0>(box) - 10);

    const int max_int0_value = 25;
    if (db::get<Int0>(box) < max_int0_value) {
      return std::tuple<decltype(std::move(box)), bool, size_t>(
          std::move(box), false,
          tmpl::index_of<ActionList, ::add_remove_test::increment_int0>::value);
    }

    CHECK(db::get<Int0>(box) == max_int0_value);
    return std::tuple<decltype(std::move(box)), bool, size_t>(
        std::move(box), true,
        tmpl::index_of<ActionList, iterate_increment_int0>::value + 1);
  }
};

struct finalize {
  template <
      typename... DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<
          tmpl2::flat_any_v<cpp17::is_same_v<CountActionsCalled, DbTags>...> and
          tmpl2::flat_any_v<cpp17::is_same_v<Int0, DbTags>...>> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    CHECK(db::get<TemporalId>(box) == db::item_type<TemporalId>{0});
    CHECK(db::get<CountActionsCalled>(box) == 31);
    CHECK(db::get<Int0>(box) == 25);
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace any_order
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.local_order", "[Unit][Parallel]") {
  // Test that out-of-order execution of Actions is supported
  Parallel::AlgorithmImpl<
      SingletonParallelComponent<TestMetavariables>,
      Parallel::Algorithms::Singleton, TestMetavariables,
      typelist<add_remove_test::add_int_value_10,
               add_remove_test::increment_int0,
               any_order::iterate_increment_int0, add_remove_test::remove_int0,
               receive_data_test::update_instance>,
      ElementId,
      db::DataBox<
          db::get_databox_list<typelist<CountActionsCalled, TemporalId>>>>
      al_gore{};
  al_gore.template explicit_single_action<add_remove_test::initialize>();
  al_gore.perform_algorithm();
  al_gore.template explicit_single_action<any_order::finalize>();
}

namespace {
struct error_size_zero {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
  }
};
}  // namespace

// [[OutputRegex, Cannot call apply function of '\(anonymous
// namespace\)::error_size_zero' with DataBox]]
SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.bad_box_apply", "[Unit][Parallel]") {
  ERROR_TEST();
  Parallel::AlgorithmImpl<SingletonParallelComponent<TestMetavariables>,
                          Parallel::Algorithms::Singleton, TestMetavariables,
                          typelist<>, ElementId,
                          db::DataBox<db::get_databox_list<typelist<>>>>
      al_gore{};
  al_gore.template explicit_single_action<error_size_zero>();
}

namespace {
template <typename Algorithm>
struct error_call_single_action_from_action {
  using apply_args = tmpl::list<Algorithm&, int>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/, Algorithm& al_gore,
                    int which) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    if (which == 0) {
      al_gore.template explicit_single_action<no_op_test::finalize>();
    }
    al_gore.template explicit_single_action<
        error_call_single_action_from_action<Parallel::AlgorithmImpl<
            SingletonParallelComponent<TestMetavariables>,
            Parallel::Algorithms::Singleton, TestMetavariables, typelist<>,
            ElementId, db::DataBox<db::get_databox_list<typelist<>>>>>>(
        std::tuple<
            Parallel::AlgorithmImpl<
                SingletonParallelComponent<TestMetavariables>,
                Parallel::Algorithms::Singleton, TestMetavariables, typelist<>,
                ElementId, db::DataBox<db::get_databox_list<typelist<>>>>&,
            int>(al_gore, 0));
  }
};
}  // namespace

// [[OutputRegex, Already performing an Action and cannot execute additional
// Actions from inside of an Action. This is only possible if the
// explicit_single_action function is not invoked via a proxy, which we do not
// allow]]
SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.action_from_action_1",
                  "[Unit][Parallel]") {
  ERROR_TEST();
  Parallel::AlgorithmImpl<SingletonParallelComponent<TestMetavariables>,
                          Parallel::Algorithms::Singleton, TestMetavariables,
                          typelist<>, ElementId,
                          db::DataBox<db::get_databox_list<typelist<>>>>
      al_gore{};
  al_gore.template explicit_single_action<
      error_call_single_action_from_action<Parallel::AlgorithmImpl<
          SingletonParallelComponent<TestMetavariables>,
          Parallel::Algorithms::Singleton, TestMetavariables, typelist<>,
          ElementId, db::DataBox<db::get_databox_list<typelist<>>>>>>(
      std::tuple<
          Parallel::AlgorithmImpl<
              SingletonParallelComponent<TestMetavariables>,
              Parallel::Algorithms::Singleton, TestMetavariables, typelist<>,
              ElementId, db::DataBox<db::get_databox_list<typelist<>>>>&,
          int>(al_gore, 0));
}

// [[OutputRegex, Already performing an Action and cannot execute additional
// Actions from inside of an Action. This is only possible if the
// explicit_single_action function is not invoked via a proxy, which we do not
// allow]]
SPECTRE_TEST_CASE("Unit.Parallel.Algorithm.action_from_action_2",
                  "[Unit][Parallel]") {
  ERROR_TEST();
  Parallel::AlgorithmImpl<SingletonParallelComponent<TestMetavariables>,
                          Parallel::Algorithms::Singleton, TestMetavariables,
                          typelist<>, ElementId,
                          db::DataBox<db::get_databox_list<typelist<>>>>
      al_gore{};
  al_gore.template explicit_single_action<
      error_call_single_action_from_action<Parallel::AlgorithmImpl<
          SingletonParallelComponent<TestMetavariables>,
          Parallel::Algorithms::Singleton, TestMetavariables, typelist<>,
          ElementId, db::DataBox<db::get_databox_list<typelist<>>>>>>(
      std::tuple<
          Parallel::AlgorithmImpl<
              SingletonParallelComponent<TestMetavariables>,
              Parallel::Algorithms::Singleton, TestMetavariables, typelist<>,
              ElementId, db::DataBox<db::get_databox_list<typelist<>>>>&,
          int>(al_gore, 1));
}

namespace {
template <class Metavariables>
struct SingletonParallelComponent {
  using chare_type = Parallel::Algorithms::Singleton;

  using metavariables = Metavariables;
  using action_list = typelist<>;
  using array_index = int;
  using initial_databox = db::DataBox<db::get_databox_list<typelist<>>>;
  using explicit_single_actions_list = tmpl::list<
      no_op_test::initialize, no_op_test::finalize, add_remove_test::initialize,
      add_remove_test::finalize, add_remove_test::test_args,
      receive_data_test::initialize, receive_data_test::finalize,
      any_order::finalize, error_size_zero,
      error_call_single_action_from_action<Parallel::AlgorithmImpl<
          SingletonParallelComponent<TestMetavariables>,
          Parallel::Algorithms::Singleton, TestMetavariables, typelist<>,
          ElementId, db::DataBox<db::get_databox_list<typelist<>>>>>>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& /*global_cache*/) {}

  static void execute_next_global_actions(
      const typename Metavariables::Phase /*next_phase*/,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) {}
};
}  // namespace
