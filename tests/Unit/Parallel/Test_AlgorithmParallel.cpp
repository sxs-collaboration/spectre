// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "AlgorithmArray.hpp"
#include "AlgorithmGroup.hpp"
#include "AlgorithmNodegroup.hpp"
#include "AlgorithmSingleton.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

static constexpr int number_of_1d_array_elements = 14;

namespace Tags {
struct Int0 : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Int0";
  using type = int;
};

struct Int1 : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Int1";
  using type = int;
};

struct CountActionsCalled : db::DataBoxTag {
  static constexpr db::DataBoxString label = "CountActionsCalled";
  using type = int;
};

struct IntReceiveTag {
  using temporal_id = int;
  using type = std::unordered_map<temporal_id, std::unordered_multiset<int>>;
};
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
  using apply_args = tmpl::list<>;

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
    return std::forward_as_tuple(box);
  }
};

struct CountReceives {
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    auto& int_receives = tuples::get<Tags::IntReceiveTag>(inboxes);
    SPECTRE_PARALLEL_REQUIRE(int_receives.size() <= 70);
    if (int_receives.size() == 70) {
      for (const auto& p : int_receives) {
        SPECTRE_PARALLEL_REQUIRE(p.second.size() == 1);
        SPECTRE_PARALLEL_REQUIRE(*(p.second.begin()) % 3 == 0);
      }
      int_receives.clear();

      // Call to arrays, have them execute once then reduce something through
      // groups and nodegroups
      auto& array_parallel_component = Parallel::get_parallel_component<
          ArrayParallelComponent<Metavariables>>(cache);
      for (int i = 0; i < number_of_1d_array_elements; ++i) {
        // we do not do a broadcast so that we can check inline entry methods on
        // array work. We pass "true" as the second argument to start the
        // algorithm up again on the arrays
        array_parallel_component[i].template receive_data<Tags::IntReceiveTag>(
            0, 101, true);
      }
    }
    return std::tuple<db::DataBox<DbTags>&&, bool>(std::move(box), true);
  }
};
}  // namespace SingletonActions

namespace ArrayActions {
struct Initialize {
  using apply_args = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(db::create<typelist<Tags::CountActionsCalled>>(0));
  }
};

struct AddIntValue10 {
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    auto& int_receives = tuples::get<Tags::IntReceiveTag>(inboxes);
    SPECTRE_PARALLEL_REQUIRE(int_receives.empty() or int_receives.size() == 1);
    if (int_receives.size() == 1) {
      auto& group_parallel_component = Parallel::get_parallel_component<
          GroupParallelComponent<Metavariables>>(cache);
      group_parallel_component.template receive_data<Tags::IntReceiveTag>(
          db::get<Tags::CountActionsCalled>(box) + 100 * array_index,
          db::get<Tags::CountActionsCalled>(box));
    }
    db::mutate<Tags::CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    const bool terminate_algorithm =
        db::get<Tags::CountActionsCalled>(box) >= 15;
    return std::make_tuple(
        db::create_from<typelist<>, typelist<Tags::Int0>>(std::move(box), 10),
        terminate_algorithm);
  }
};

struct IncrementInt0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    db::mutate<Tags::CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    db::mutate<Tags::Int0>(box, [](int& int0) { int0++; });
    return std::forward_as_tuple(std::move(box));
  }
};

struct RemoveInt0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<Tags::Int0>(box) == 11);
    db::mutate<Tags::CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    return std::make_tuple(
        db::create_from<typelist<Tags::Int0>>(std::move(box)));
  }
};

struct SendToSingleton {
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    auto& singleton_parallel_component = Parallel::get_parallel_component<
        SingletonParallelComponent<Metavariables>>(cache);
    // Send CountActionsCalled to the SingletonParallelComponent several times
    singleton_parallel_component.template receive_data<Tags::IntReceiveTag>(
        db::get<Tags::CountActionsCalled>(box) + 100 * array_index,
        db::get<Tags::CountActionsCalled>(box), true);
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace ArrayActions

namespace GroupActions {
struct Initialize {
  using apply_args = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   GroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(db::create<typelist<Tags::CountActionsCalled>>(0));
  }
};

struct PrintSomething {
  using inbox_tags = tmpl::list<Tags::IntReceiveTag>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         GroupParallelComponent<TestMetavariables>> or
            cpp17::is_same_v<ParallelComponent,
                             NodegroupParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool>(
        std::move(box), true);
  }
};

struct ReduceInt {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   GroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    return std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool>(
        std::move(box), true);
  }
};

}  // namespace GroupActions

namespace NodegroupActions {
struct Initialize {
  using apply_args = tmpl::list<>;

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
                         NodegroupParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(db::create<typelist<Tags::CountActionsCalled>>(0));
  }
};

}  // namespace NodegroupActions

/// [singleton_parallel_component]
template <class Metavariables>
struct SingletonParallelComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = typelist<>;
  using metavariables = Metavariables;
  using action_list = typelist<SingletonActions::CountReceives>;
  using initial_databox = db::DataBox<db::get_databox_list<typelist<>>>;
  using explicit_single_actions_list = tmpl::list<SingletonActions::Initialize>;
  using options = typelist<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<SingletonParallelComponent>(local_cache)
        .template explicit_single_action<SingletonActions::Initialize>();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    if (next_phase == Metavariables::Phase::PerformSingletonAlgorithm) {
      auto& local_cache = *(global_cache.ckLocalBranch());
      Parallel::get_parallel_component<SingletonParallelComponent>(local_cache)
          .perform_algorithm();
      return;
    }
  }
};
/// [singleton_parallel_component]

/// [array_parallel_component]
template <class Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using const_global_cache_tag_list = typelist<>;
  using metavariables = Metavariables;
  using action_list =
      typelist<ArrayActions::AddIntValue10, ArrayActions::IncrementInt0,
               ArrayActions::RemoveInt0, ArrayActions::SendToSingleton>;
  using array_index = int;
  using initial_databox =
      db::DataBox<db::get_databox_list<typelist<Tags::CountActionsCalled>>>;

  using explicit_single_actions_list = tmpl::list<ArrayActions::Initialize>;
  using options = typelist<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(local_cache);

    for (int i = 0, which_proc = 0,
             number_of_procs = Parallel::number_of_procs();
         i < number_of_1d_array_elements; ++i) {
      array_proxy[i].insert(global_cache, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
    array_proxy.doneInserting();

    array_proxy.template explicit_single_action<ArrayActions::Initialize>();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::PerformArrayAlgorithm) {
      Parallel::get_parallel_component<ArrayParallelComponent>(local_cache)
          .perform_algorithm();
    }
  }
};
/// [array_parallel_component]

template <class Metavariables>
struct GroupParallelComponent {
  using chare_type = Parallel::Algorithms::Group;
  using const_global_cache_tag_list = typelist<>;
  using metavariables = Metavariables;
  using action_list = typelist<GroupActions::PrintSomething>;
  using initial_databox =
      db::DataBox<db::get_databox_list<typelist<Tags::CountActionsCalled>>>;

  using explicit_single_actions_list = tmpl::list<GroupActions::Initialize>;
  using options = typelist<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<GroupParallelComponent>(local_cache)
        .template explicit_single_action<GroupActions::Initialize>();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& /*global_cache*/) {}
};

template <class Metavariables>
struct NodegroupParallelComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tag_list = typelist<>;
  using metavariables = Metavariables;
  using action_list = typelist<GroupActions::PrintSomething>;
  using initial_databox =
      db::DataBox<db::get_databox_list<typelist<Tags::CountActionsCalled>>>;

  using explicit_single_actions_list =
      tmpl::list<NodegroupActions::Initialize>;
  using options = typelist<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<NodegroupParallelComponent>(local_cache)
        .template explicit_single_action<NodegroupActions::Initialize>();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& /*global_cache*/) {}
};

struct TestMetavariables {
  using component_list =
      tmpl::list<SingletonParallelComponent<TestMetavariables>,
                 ArrayParallelComponent<TestMetavariables>,
                 GroupParallelComponent<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;

  static constexpr const char* const help{"Test Algorithm in parallel"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  enum class Phase {
    Initialization,
    PerformSingletonAlgorithm,
    PerformArrayAlgorithm,
    Run2,
    Exit
  };
  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        TestMetavariables>& /*cache_proxy*/) {
    Parallel::printf("Determining next phase\n");

    if (current_phase == Phase::Initialization) {
      return Phase::PerformSingletonAlgorithm;
    } else if (current_phase == Phase::PerformSingletonAlgorithm) {
      return Phase::PerformArrayAlgorithm;
    } else if (current_phase == Phase::PerformArrayAlgorithm) {
      // TODO(nils): check if Run2 should do something, otherwise remove it
      return Phase::Run2;
    }

    return Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{};

using charm_metavariables = TestMetavariables;

#include "Parallel/CharmMain.cpp"
