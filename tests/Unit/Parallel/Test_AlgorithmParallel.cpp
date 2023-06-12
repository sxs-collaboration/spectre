// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
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
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

static constexpr int number_of_1d_array_elements = 14;

namespace PUP {
class er;
}  // namespace PUP
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

namespace AlgorithmParallel_detail {
struct UnpackCounter {
  UnpackCounter() = default;
  ~UnpackCounter() = default;
  UnpackCounter(const UnpackCounter& /*unused*/) = default;
  UnpackCounter& operator=(const UnpackCounter& /*unused*/) = default;
  UnpackCounter(UnpackCounter&& /*unused*/) = default;
  UnpackCounter& operator=(UnpackCounter&& /*unused*/) = default;

  explicit UnpackCounter(CkMigrateMessage* /*msg*/) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | counter_value;
    if (p.isUnpacking()) {
      ++counter_value;
    }
  }
  size_t counter_value = 0;
};
}  // namespace AlgorithmParallel_detail

namespace Tags {
struct ReceiveArrayComponentsOnceMore : db::SimpleTag {
  using type = bool;
};

struct Int0 : db::SimpleTag {
  static std::string name() { return "Int0"; }
  using type = int;
};

struct Int1 : db::SimpleTag {
  static std::string name() { return "Int1"; }
  using type = int;
};

struct CountActionsCalled : db::SimpleTag {
  static std::string name() { return "CountActionsCalled"; }
  using type = int;
};

struct UnpackCounter : db::SimpleTag {
  using type = AlgorithmParallel_detail::UnpackCounter;
};

// [int_receive_tag]
struct IntReceiveTag {
  using temporal_id = int;
  using type = std::unordered_map<temporal_id, std::unordered_multiset<int>>;

  template <typename Inbox, typename ReceiveDataType>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                const temporal_id& temporal_id_v,
                                ReceiveDataType&& data) {
    (*inbox)[temporal_id_v].insert(std::forward<ReceiveDataType>(data));
  }
};
// [int_receive_tag]
}  // namespace Tags

struct TestMetavariables;

template <class Metavariables>
struct SingletonParallelComponent;

template <class Metavariables>
struct ArrayParallelComponent;

template <class Metavariables>
struct GroupParallelComponent;

template <class Metavariables>
struct NodegroupParallelComponent;

namespace SingletonActions {

struct Initialize {
  using simple_tags = tmpl::list<Tags::ReceiveArrayComponentsOnceMore>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 SingletonParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), false);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

// One component (A) may call this simple action on another component (B) so
// that B then invokes the `perform_algorithm` entry method on A, which will
// restart A if it has been terminated.
// This helps to add artificial breaks to the iterable actions to better test
// charm runtime processes that must wait for the components to be outside entry
// methods, such as load balancing.
template <typename ComponentToRestart>
struct RestartMe {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename IndexToRestart, typename Metavariables>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const IndexToRestart index_to_restart) {
    Parallel::get_parallel_component<ComponentToRestart>(
        cache)[index_to_restart]
        .perform_algorithm(true);
  }
};

struct CountReceives {
  // [int_receive_tag_list]
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;
  // [int_receive_tag_list]

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 SingletonParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    auto& int_receives = tuples::get<Tags::IntReceiveTag>(inboxes);
    if (db::get<Tags::ReceiveArrayComponentsOnceMore>(box)) {
      SPECTRE_PARALLEL_REQUIRE(int_receives.size() <= 14);
      if (int_receives.size() != 14) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
    } else {
      SPECTRE_PARALLEL_REQUIRE(int_receives.size() <= 686);
      if (int_receives.size() != 686) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
    }

    for (const auto& p : int_receives) {
      SPECTRE_PARALLEL_REQUIRE(p.second.size() == 1);
      SPECTRE_PARALLEL_REQUIRE(*(p.second.begin()) % 3 == 0);
    }
    int_receives.clear();

    if (not db::get<Tags::ReceiveArrayComponentsOnceMore>(box)) {
      // Call to arrays, have them execute once then reduce something through
      // groups and nodegroups
      // We do not do a broadcast so that we can check inline entry methods on
      // array work. We pass "true" as the second argument to start the
      // algorithm up again on the arrays
      // [call_on_indexed_array]
      auto& array_parallel_component = Parallel::get_parallel_component<
          ArrayParallelComponent<Metavariables>>(cache);
      for (int i = 0; i < number_of_1d_array_elements; ++i) {
        Parallel::receive_data<Tags::IntReceiveTag>(array_parallel_component[i],
                                                    0, 101, true);
      }
      // [call_on_indexed_array]
    }

    db::mutate<Tags::ReceiveArrayComponentsOnceMore>(
        [](bool* const receive_array_components_once_more) {
          *receive_array_components_once_more = true;
        },
        make_not_null(&box));

    // [return_with_termination]
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
    // [return_with_termination]
  }
};
}  // namespace SingletonActions

namespace ArrayActions {
struct Initialize {
  using simple_tags = tmpl::list<Tags::CountActionsCalled, Tags::UnpackCounter>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), 0, AlgorithmParallel_detail::UnpackCounter{});
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct AddIntValue10 {
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;
  using simple_tags = tmpl::list<Tags::Int0>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    auto& int_receives = tuples::get<Tags::IntReceiveTag>(inboxes);
    SPECTRE_PARALLEL_REQUIRE(int_receives.empty() or int_receives.size() == 1);
    if (int_receives.size() == 1) {
      // [broadcast_to_group]
      auto& group_parallel_component = Parallel::get_parallel_component<
          GroupParallelComponent<Metavariables>>(cache);
      Parallel::receive_data<Tags::IntReceiveTag>(
          group_parallel_component,
          db::get<Tags::CountActionsCalled>(box) + 100 * array_index,
          db::get<Tags::CountActionsCalled>(box));
      // [broadcast_to_group]
    }
    db::mutate<Tags::CountActionsCalled>(
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 10);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct CheckWasUnpacked {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Check to be sure the algorithm has been packed and unpacked at least a
    // few times during the algorithm and retained functionality
    if (sys::number_of_procs() > 1) {
      // If this check fails with slightly too few unpacks counted, it may
      // indicate that the test machine is too fast for the setting used in
      // the balancer in the accompanying CMakeLists.txt. If that is the
      // problem, you might solve it either by balancing even more often, or by
      // doing more computational work during each iteration of the algorithm.
      SPECTRE_PARALLEL_REQUIRE(db::get<Tags::UnpackCounter>(box).counter_value >
                               2);
    }
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct IncrementInt0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    db::mutate<Tags::CountActionsCalled>(
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));
    db::mutate<Tags::Int0>([](const gsl::not_null<int*> int0) { ++*int0; },
                           make_not_null(&box));
    // [iterable_action_return_continue_next_action]
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    // [iterable_action_return_continue_next_action]
  }
};

struct RemoveInt0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<Tags::Int0>(box) == 11);
    db::mutate<Tags::CountActionsCalled>(
        [](const gsl::not_null<int*> count_actions_called) {
          ++*count_actions_called;
        },
        make_not_null(&box));
    // Run the iterable action sequence several times to ensure that
    // load-balancing has a chance to be invoked multiple times.
    if (db::get<Tags::CountActionsCalled>(box) < 150) {
      // The simple action that runs on the singleton will invoke the
      // `perform_algorithm` entry method on the present component, restarting
      // it.
      // Our use of the charm runtime system ensures that the return value of
      // this iterable action will be processed before that entry method, and
      // that the QD will not trigger when the entry method is waiting to be
      // run.
      Parallel::simple_action<SingletonActions::RestartMe<ParallelComponent>>(
          Parallel::get_parallel_component<
              SingletonParallelComponent<Metavariables>>(cache),
          array_index);
    }
    // default assign Int0 to "remove" it
    Initialization::mutate_assign<tmpl::list<Tags::Int0>>(make_not_null(&box),
                                                          0);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct SendToSingleton {
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    auto& singleton_parallel_component = Parallel::get_parallel_component<
        SingletonParallelComponent<Metavariables>>(cache);
    // Send CountActionsCalled to the SingletonParallelComponent several times
    // [receive_broadcast]
    Parallel::receive_data<Tags::IntReceiveTag>(
        singleton_parallel_component,
        db::get<Tags::CountActionsCalled>(box) + 1000 * array_index,
        db::get<Tags::CountActionsCalled>(box), true);
    // [receive_broadcast]
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace ArrayActions

namespace GroupActions {
struct Initialize {
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;
  using simple_tags = tmpl::list<Tags::CountActionsCalled>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 GroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 0);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct CheckComponentType {
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        std::is_same_v<ParallelComponent,
                       GroupParallelComponent<TestMetavariables>> or
            std::is_same_v<ParallelComponent,
                           NodegroupParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct ReduceInt {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 GroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

}  // namespace GroupActions

namespace NodegroupActions {
struct Initialize {
  using simple_tags = tmpl::list<Tags::CountActionsCalled>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 NodegroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 0);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

}  // namespace NodegroupActions

// [singleton_parallel_component]
template <class Metavariables>
struct SingletonParallelComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<SingletonActions::Initialize>>,
      Parallel::PhaseActions<Parallel::Phase::Solve,
                             tmpl::list<SingletonActions::CountReceives>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<SingletonParallelComponent>(local_cache)
        .start_phase(next_phase);
  }
};
// [singleton_parallel_component]

// [array_parallel_component]
template <class Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ArrayActions::Initialize>>,
      Parallel::PhaseActions<
          Parallel::Phase::Solve,
          tmpl::list<ArrayActions::AddIntValue10, ArrayActions::IncrementInt0,
                     ArrayActions::RemoveInt0, ArrayActions::SendToSingleton>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<ArrayActions::CheckWasUnpacked>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  using array_index = int;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(local_cache);

    TestHelpers::Parallel::assign_array_elements_round_robin_style(
        array_proxy, static_cast<size_t>(number_of_1d_array_elements),
        static_cast<size_t>(sys::number_of_procs()), {}, global_cache,
        procs_to_ignore);
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<ArrayParallelComponent>(local_cache)
        .start_phase(next_phase);
  }
};
// [array_parallel_component]

template <class Metavariables>
struct GroupParallelComponent {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<GroupActions::Initialize, GroupActions::CheckComponentType>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase /*next_phase*/,
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) {}
};

template <class Metavariables>
struct NodegroupParallelComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<NodegroupActions::Initialize,
                                        GroupActions::CheckComponentType>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase /*next_phase*/,
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) {}
};

struct TestMetavariables {
  using component_list =
      tmpl::list<SingletonParallelComponent<TestMetavariables>,
                 ArrayParallelComponent<TestMetavariables>,
                 GroupParallelComponent<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;

  static constexpr const char* const help{"Test Algorithm in parallel"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Solve,
       Parallel::Phase::Testing, Parallel::Phase::Cleanup,
       Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

// [charm_include_example]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
// [charm_include_example]
