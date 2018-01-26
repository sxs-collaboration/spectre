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

static constexpr int number_of_1d_array_elements_per_core = 10;

struct TestMetavariables;

template <class Metavariables>
struct ArrayParallelComponent;

template <class Metavariables>
struct NodegroupParallelComponent;

namespace Tags {
struct vector_of_array_indexs : db::DataBoxTag {
  static constexpr db::DataBoxString label = "vector_of_array_indexs";
  using type = std::vector<int>;
};

struct total_receives_on_node : db::DataBoxTag {
  static constexpr db::DataBoxString label = "total_receives_on_node";
  using type = int;
};
}  // namespace Tags

struct nodegroup_initialize {
  using apply_args = tmpl::list<>;

  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         NodegroupParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(db::create<typelist<Tags::vector_of_array_indexs,
                                               Tags::total_receives_on_node>>(
        std::vector<int>(
            static_cast<size_t>(number_of_1d_array_elements_per_core *
                                Parallel::procs_on_node(Parallel::my_node()))),
        0));
  }
};

struct nodegroup_receive {
  using apply_args = tmpl::list<int>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) == 2> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const int& id_of_array) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         NodegroupParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<Tags::vector_of_array_indexs, Tags::total_receives_on_node>(
        box, [&id_of_array](std::vector<int>& array_indexs,
                            int& total_receives_on_node) {
          if (static_cast<int>(array_indexs.size()) !=
              number_of_1d_array_elements_per_core *
                  Parallel::procs_on_node(Parallel::my_node())) {
            array_indexs[static_cast<size_t>(id_of_array)]++;
          }
          std::for_each(array_indexs.begin(), array_indexs.end(),
                        [](int& t) { t++; });
          total_receives_on_node++;
        });
    return std::forward_as_tuple(box);
  }
};

struct nodegroup_check_first_result {
  using apply_args = tmpl::list<>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) == 2> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         NodegroupParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<Tags::total_receives_on_node>(box) ==
                             number_of_1d_array_elements_per_core *
                                 Parallel::procs_on_node(Parallel::my_node()));
    decltype(auto) v = db::get<Tags::vector_of_array_indexs>(box);
    SPECTRE_PARALLEL_REQUIRE(static_cast<int>(v.size()) ==
                             number_of_1d_array_elements_per_core *
                                 Parallel::procs_on_node(Parallel::my_node()));
    std::for_each(v.begin(), v.end(), [](const int& value) {
      SPECTRE_PARALLEL_REQUIRE(
          value == number_of_1d_array_elements_per_core *
                       Parallel::procs_on_node(Parallel::my_node()));
    });
    return std::forward_as_tuple(std::move(box));
  }
};

struct nodegroup_threaded_receive {
  using apply_args = tmpl::list<>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename NodeLock,
            Requires<sizeof...(DbTags) == 2> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const NodeLock& node_lock, const int& id_of_array) {
    Parallel::lock(node_lock);
    db::mutate<Tags::vector_of_array_indexs, Tags::total_receives_on_node>(
        box, [&id_of_array](std::vector<int>& array_indexs,
                            int& total_receives_on_node) {
          if (static_cast<int>(array_indexs.size()) !=
              number_of_1d_array_elements_per_core *
                  Parallel::procs_on_node(Parallel::my_node())) {
            array_indexs[static_cast<size_t>(id_of_array)]++;
          }
          std::for_each(array_indexs.begin(), array_indexs.end(),
                        [](int& t) { t++; });
          total_receives_on_node++;
        });
    Parallel::unlock(node_lock);
  }
};

struct nodegroup_check_threaded_result {
  using apply_args = tmpl::list<>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) == 2> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         NodegroupParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<Tags::total_receives_on_node>(box) ==
                             2 * number_of_1d_array_elements_per_core *
                                 Parallel::procs_on_node(Parallel::my_node()));
    decltype(auto) v = db::get<Tags::vector_of_array_indexs>(box);
    SPECTRE_PARALLEL_REQUIRE(static_cast<int>(v.size()) ==
                             number_of_1d_array_elements_per_core *
                                 Parallel::procs_on_node(Parallel::my_node()));
    std::for_each(v.begin(), v.end(), [](const int& value) {
      SPECTRE_PARALLEL_REQUIRE(
          value == 2 * number_of_1d_array_elements_per_core *
                       Parallel::procs_on_node(Parallel::my_node()));
    });
    return std::forward_as_tuple(std::move(box));
  }
};

struct reduce_to_nodegroup {
  using apply_args = tmpl::list<>;

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
    auto& local_nodegroup =
        *(Parallel::get_parallel_component<
              NodegroupParallelComponent<Metavariables>>(cache)
              .ckLocalBranch());
    local_nodegroup.template explicit_single_action<nodegroup_receive>(
        std::make_tuple(array_index));
    return std::forward_as_tuple(std::move(box));
  }
};

struct reduce_threaded_method {
  using apply_args = tmpl::list<>;

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
    auto& local_nodegroup =
        *(Parallel::get_parallel_component<
              NodegroupParallelComponent<Metavariables>>(cache)
              .ckLocalBranch());
    local_nodegroup.template threaded_single_action<nodegroup_threaded_receive>(
        array_index);
    return std::forward_as_tuple(std::move(box));
  }
};

template <class Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using const_global_cache_tag_list = typelist<>;
  using options = typelist<>;
  using metavariables = Metavariables;
  using action_list = typelist<>;
  using array_index = int;
  using initial_databox = db::DataBox<typelist<>>;

  using explicit_single_actions_list =
      tmpl::list<reduce_to_nodegroup, reduce_threaded_method>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(local_cache);

    int which_proc = 0;
    const int number_of_procs = Parallel::number_of_procs();
    for (int i = 0;
         i < number_of_1d_array_elements_per_core * Parallel::number_of_procs();
         ++i) {
      array_proxy[i].insert(global_cache, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
    array_proxy.doneInserting();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(local_cache);
    if (next_phase == Metavariables::Phase::ArrayToNodegroup) {
      array_proxy.template explicit_single_action<reduce_to_nodegroup>();
    }
    if (next_phase == Metavariables::Phase::TestThreadedMethod) {
      array_proxy.template explicit_single_action<reduce_threaded_method>();
    }
  }
};

template <class Metavariables>
struct NodegroupParallelComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tag_list = typelist<>;
  using options = typelist<>;
  using metavariables = Metavariables;
  using action_list = typelist<>;
  using initial_databox = db::DataBox<db::get_databox_list<
      typelist<Tags::total_receives_on_node, Tags::vector_of_array_indexs>>>;

  using explicit_single_actions_list =
      tmpl::list<nodegroup_initialize, nodegroup_receive,
                 nodegroup_check_first_result, nodegroup_check_threaded_result>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<NodegroupParallelComponent>(local_cache)
        .template explicit_single_action<nodegroup_initialize>();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& nodegroup_proxy =
        Parallel::get_parallel_component<NodegroupParallelComponent>(
            local_cache);
    if (next_phase == Metavariables::Phase::CheckFirstResult) {
      nodegroup_proxy
          .template explicit_single_action<nodegroup_check_first_result>();
    }
    if (next_phase == Metavariables::Phase::CheckThreadedResult) {
      nodegroup_proxy
          .template explicit_single_action<nodegroup_check_threaded_result>();
    }
  }
};

struct TestMetavariables {
  using component_list =
      tmpl::list<ArrayParallelComponent<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;

  static constexpr const char* const help{"Test nodelocks in Algorithm"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  enum class Phase {
    Initialization,
    ArrayToNodegroup,
    CheckFirstResult,
    TestThreadedMethod,
    CheckThreadedResult,
    Exit
  };
  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        TestMetavariables>& /*cache_proxy*/) {
    Parallel::printf("Determining next phase\n");

    if (current_phase == Phase::Initialization) {
      return Phase::ArrayToNodegroup;
    }
    if (current_phase == Phase::ArrayToNodegroup) {
      return Phase::CheckFirstResult;
    }
    if (current_phase == Phase::CheckFirstResult) {
      return Phase::TestThreadedMethod;
    }
    if (current_phase == Phase::TestThreadedMethod) {
      return Phase::CheckThreadedResult;
    }

    return Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{};

using charm_metavariables = TestMetavariables;

#include "Parallel/CharmMain.cpp"
