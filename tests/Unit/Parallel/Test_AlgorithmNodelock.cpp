// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "AlgorithmArray.hpp"
#include "AlgorithmNodegroup.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Parallel/Algorithm.hpp"  // IWYU pragma: keep
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

static constexpr int number_of_1d_array_elements_per_core = 10;

struct TestMetavariables;

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

template <class Metavariables>
struct ArrayParallelComponent;

template <class Metavariables>
struct NodegroupParallelComponent;

namespace Tags {
struct vector_of_array_indexs : db::SimpleTag {
  static std::string name() noexcept { return "vector_of_array_indexs"; }
  using type = std::vector<int>;
};

struct total_receives_on_node : db::SimpleTag {
  static std::string name() noexcept { return "total_receives_on_node"; }
  using type = int;
};
}  // namespace Tags

struct nodegroup_initialize {
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
    return std::make_tuple(db::create<tmpl::list<Tags::vector_of_array_indexs,
                                                 Tags::total_receives_on_node>>(
        std::vector<int>(
            static_cast<size_t>(number_of_1d_array_elements_per_core *
                                Parallel::procs_on_node(Parallel::my_node()))),
        0));
  }
};

struct nodegroup_receive {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) == 2> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
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
        make_not_null(&box),
        [&id_of_array](const gsl::not_null<std::vector<int>*> array_indexs,
                       const gsl::not_null<int*> total_receives_on_node) {
          if (static_cast<int>(array_indexs->size()) !=
              number_of_1d_array_elements_per_core *
                  Parallel::procs_on_node(Parallel::my_node())) {
            (*array_indexs)[static_cast<size_t>(id_of_array)]++;
          }
          std::for_each(array_indexs->begin(), array_indexs->end(),
                        [](int& t) { t++; });
          ++*total_receives_on_node;
        });
  }
};

struct nodegroup_check_first_result {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) == 2> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
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
  }
};

struct nodegroup_threaded_receive {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, typename NodeLock,
            Requires<sizeof...(DbTags) == 2> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const gsl::not_null<NodeLock*> node_lock,
                    const int& id_of_array) {
    Parallel::lock(node_lock);
    db::mutate<Tags::vector_of_array_indexs, Tags::total_receives_on_node>(
        make_not_null(&box),
        [&id_of_array](const gsl::not_null<std::vector<int>*> array_indexs,
                       const gsl::not_null<int*> total_receives_on_node) {
          if (static_cast<int>(array_indexs->size()) !=
              number_of_1d_array_elements_per_core *
                  Parallel::procs_on_node(Parallel::my_node())) {
            (*array_indexs)[static_cast<size_t>(id_of_array)]++;
          }
          std::for_each(array_indexs->begin(), array_indexs->end(),
                        [](int& t) { t++; });
          ++*total_receives_on_node;
        });
    Parallel::unlock(node_lock);
  }
};

struct nodegroup_check_threaded_result {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) == 2> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
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
  }
};

struct reduce_to_nodegroup {
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
    /// [simple_action_with_args]
    Parallel::simple_action<nodegroup_receive>(local_nodegroup, array_index);
    /// [simple_action_with_args]
    return std::forward_as_tuple(std::move(box));
  }
};

struct reduce_threaded_method {
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
    Parallel::threaded_action<nodegroup_threaded_receive>(local_nodegroup,
                                                          array_index);
    return std::forward_as_tuple(std::move(box));
  }
};

template <class Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using array_index = int;
  using initial_databox = db::compute_databox_type<tmpl::list<>>;

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

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(local_cache);
    if (next_phase == Metavariables::Phase::ArrayToNodegroup) {
      Parallel::simple_action<reduce_to_nodegroup>(array_proxy);
    }
    if (next_phase == Metavariables::Phase::TestThreadedMethod) {
      Parallel::simple_action<reduce_threaded_method>(array_proxy);
    }
  }
};

template <class Metavariables>
struct NodegroupParallelComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      tmpl::list<Tags::vector_of_array_indexs, Tags::total_receives_on_node>>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<nodegroup_initialize>(
        Parallel::get_parallel_component<NodegroupParallelComponent>(
            local_cache));
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& nodegroup_proxy =
        Parallel::get_parallel_component<NodegroupParallelComponent>(
            local_cache);
    if (next_phase == Metavariables::Phase::CheckFirstResult) {
      Parallel::simple_action<nodegroup_check_first_result>(nodegroup_proxy);
    }
    if (next_phase == Metavariables::Phase::CheckThreadedResult) {
      Parallel::simple_action<nodegroup_check_threaded_result>(nodegroup_proxy);
    }
  }
};

struct TestMetavariables {
  using component_list =
      tmpl::list<ArrayParallelComponent<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

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
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
