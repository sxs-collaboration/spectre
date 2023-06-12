// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
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

template <size_t Dim>
using BoundaryMessage = evolution::dg::BoundaryMessage<Dim>;

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

struct TemporalId0 : db::SimpleTag {
  static std::string name() { return "TemporalId0"; }
  using type = TestAlgorithmArrayInstance;
};

struct TemporalId1 : db::SimpleTag {
  using type = ::TimeStepId;
};

struct Vector0 : db::SimpleTag {
  using type = DataVector;
};

struct Vector1 : db::SimpleTag {
  using type = DataVector;
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
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));
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
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));
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
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));
    db::mutate<Int0>([](const gsl::not_null<int*> int0) { ++*int0; },
                     make_not_null(&box));
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
        [](const gsl::not_null<int*> count_actions_called, const int& int0) {
          SPECTRE_PARALLEL_REQUIRE(int0 == 11);
          ++*count_actions_called;
        },
        make_not_null(&box), db::get<Int0>(box));
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
  using simple_tags = tmpl::list<CountActionsCalled, TemporalId0>;
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

struct BoundaryMessageReceiveTag {
  using temporal_id = ::TimeStepId;
  using type =
      std::unordered_map<temporal_id,
                         std::map<std::pair<Direction<3>, ElementId<3>>,
                                  std::unique_ptr<BoundaryMessage<3>>>>;
  using message_type = BoundaryMessage<3>;

  template <typename Inbox>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                BoundaryMessage<3>* boundary_message) {
    auto& time_step_id = boundary_message->current_time_step_id;
    auto& current_inbox = (*inbox)[time_step_id];

    auto key = std::make_pair(boundary_message->neighbor_direction,
                              boundary_message->element_id);

    current_inbox.insert_or_assign(
        key, std::unique_ptr<BoundaryMessage<3>>(boundary_message));
  }
};

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

struct add_vectors_to_box_and_send {
  using simple_tags = tmpl::list<TemporalId1, Vector0, Vector1>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    TimeStepId time_step_id{true, 0, Slab{0.0, 1.1}.start()};

    db::mutate<TemporalId1, Vector0, Vector1>(
        [&time_step_id](const gsl::not_null<TimeStepId*> time_step_id_ptr,
                        const gsl::not_null<DataVector*> vector_0_ptr,
                        const gsl::not_null<DataVector*> vector_1_ptr) {
          *time_step_id_ptr = time_step_id;
          *vector_0_ptr = DataVector{-4.6, 9.8, 3.6, -1.7};
          *vector_1_ptr = DataVector{};
        },
        make_not_null(&box));

    BoundaryMessage<3>* boundary_message = new BoundaryMessage<3>(
        0, 4, false, true, Parallel::my_node<size_t>(cache),
        Parallel::my_proc<size_t>(cache), -2, time_step_id, time_step_id, {},
        {}, {}, {}, nullptr, const_cast<double*>(db::get<Vector0>(box).data()));

    // Send to myself because everybody is on the same node and it's easier to
    // check that pointers are the same between my own Vector0 and Vector1 than
    // a different element. The pointer still goes through charm either way.
    Parallel::receive_data<receive_data_test::BoundaryMessageReceiveTag>(
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        boundary_message);

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
    db::mutate<Int1>([](const gsl::not_null<int*> int1) { ++*int1; },
                     make_not_null(&box));
    // [retry_example]
    if (inbox.count(db::get<TemporalId0>(box)) == 0) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    // [retry_example]

    db::mutate<CountActionsCalled>(
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));
    static int a = 0;
    auto int0 = *inbox[db::get<TemporalId0>(box)].begin();
    inbox.erase(db::get<TemporalId0>(box));
    db::mutate<Int0>(
        [&int0](const gsl::not_null<int*> int0_box) { *int0_box = int0; },
        make_not_null(&box));
    return {++a >= 5 ? Parallel::AlgorithmExecution::Pause
                     : Parallel::AlgorithmExecution::Continue,
            std::nullopt};
  }
};

struct set_vector1_from_receive {
  using inbox_tags = tmpl::list<BoundaryMessageReceiveTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<BoundaryMessageReceiveTag>(inboxes);

    if (inbox.count(db::get<TemporalId1>(box)) == 0) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    db::mutate<CountActionsCalled>(
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));

    auto& message_map = inbox[db::get<TemporalId1>(box)];
    // We only sent one message so there should only be one in the inbox
    SPECTRE_PARALLEL_REQUIRE(message_map.size() == 1);

    auto& boundary_message = message_map.begin()->second;

    // Set the data reference
    db::mutate<Vector1>(
        [&boundary_message](const gsl::not_null<DataVector*> vector1_box) {
          vector1_box->set_data_ref(boundary_message->dg_flux_data,
                                    boundary_message->dg_flux_data_size);
        },
        make_not_null(&box));

    // We shouldn't have gone through the boundary_message::pack() function, so
    // this shouldn't be true
    SPECTRE_PARALLEL_REQUIRE_FALSE(boundary_message->owning);

    // Only the boundary message gets destroyed, not the data it points to
    inbox.erase(db::get<TemporalId1>(box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
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
    db::mutate<TemporalId0>(
        [](const gsl::not_null<TestAlgorithmArrayInstance*> temporal_id) {
          ++*temporal_id;
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct initialize {
  using simple_tags =
      tmpl::list<CountActionsCalled, Int1, TemporalId0, TemporalId1>;
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
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), 0, 0, TestAlgorithmArrayInstance{0}, {});
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
    SPECTRE_PARALLEL_REQUIRE(db::get<TemporalId0>(box) ==
                             TestAlgorithmArrayInstance{4});
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 14);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int1>(box) == 10);

    // Check that the data itself is equal
    SPECTRE_PARALLEL_REQUIRE(db::get<Vector0>(box) == db::get<Vector1>(box));
    // Now check that the pointers are equal, because they should be
    SPECTRE_PARALLEL_REQUIRE(db::get<Vector0>(box).data() ==
                             db::get<Vector1>(box).data());
  }
};
}  // namespace receive_data_test

template <class Metavariables>
struct ReceiveComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = ElementId<3>;  // Just to test nothing breaks
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<receive_data_test::initialize>>,
      Parallel::PhaseActions<
          Parallel::Phase::ImportInitialData,
          tmpl::list<receive_data_test::add_int0_to_box,
                     receive_data_test::set_int0_from_receive,
                     add_remove_test::increment_int0,
                     add_remove_test::remove_int0,
                     receive_data_test::update_instance>>,
      Parallel::PhaseActions<
          Parallel::Phase::AdjustDomain,
          tmpl::list<receive_data_test::add_vectors_to_box_and_send,
                     receive_data_test::set_vector1_from_receive,
                     Parallel::Actions::TerminatePhase>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_items,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    TestHelpers::Parallel::assign_array_elements_round_robin_style(
        Parallel::get_parallel_component<ReceiveComponent>(
            *Parallel::local_branch(global_cache)),
        1, 1, initialization_items, global_cache, procs_to_ignore,
        ElementId<3>{0});
  }

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
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));
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
    SPECTRE_PARALLEL_REQUIRE(db::get<TemporalId0>(box) ==
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

  // These phases are just here to separate out execution of the test. The names
  // of the phases don't correspond to what actually happens in the test except
  // for Initialization and Exit. The rest are simply used as individual
  // sections to test a particular feature.
  static constexpr std::array<Parallel::Phase, 11> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::Testing, Parallel::Phase::Solve,
       Parallel::Phase::Evolve, Parallel::Phase::ImportInitialData,
       Parallel::Phase::AdjustDomain,
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
    &enable_floating_point_exceptions, &enable_segfault_handler};
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
