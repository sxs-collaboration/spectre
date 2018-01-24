// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <unordered_map>

#include "AlgorithmArray.hpp"
#include "AlgorithmGroup.hpp"
#include "AlgorithmNodegroup.hpp"
#include "AlgorithmSingleton.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

struct TestMetavariables;
template <class Metavariables>
struct SingletonParallelComponent;

static constexpr int number_of_1d_array_elements = 14;

/// [custom_reduce_function]
CkReductionMsg* reduce_reduction_data(const int number_of_messages,
                                      CkReductionMsg** const msgs) {
  // clang-tidy: do not use pointer arithmetic
  Parallel::ReductionData<int, std::unordered_map<std::string, int>,
                          std::vector<int>>
      reduced(msgs[0]);  // NOLINT
  for (int msg_id = 1; msg_id < number_of_messages; ++msg_id) {
    // clang-tidy: do not use pointer arithmetic
    Parallel::ReductionData<int, std::unordered_map<std::string, int>,
                            std::vector<int>>
        current(msgs[msg_id]);  // NOLINT
    if (Parallel::get<0>(current) != Parallel::get<0>(reduced)) {
      Parallel::abort("Tried to reduce from different iteration values.");
    }
    // compute maximum of each value in an unordered_map
    for (const auto& string_double : Parallel::get<1>(current)) {
      if (string_double.second >
          Parallel::get<1>(reduced)[string_double.first]) {
        Parallel::get<1>(reduced)[string_double.first] = string_double.second;
      }
    }
    // compute sum of each element in a vector
    for (size_t i = 0; i < Parallel::get<2>(current).size(); ++i) {
      Parallel::get<2>(reduced)[i] += Parallel::get<2>(current)[i];
    }
  }
  return Parallel::new_reduction_msg(reduced);
}
/// [custom_reduce_function]

/// [custom_reduce_register_function]
CkReduction::reducerType reduction_data_reducer;

void register_custom_reductions() {
  reduction_data_reducer = CkReduction::addReducer(reduce_reduction_data);
}
/// [custom_reduce_register_function]

struct singleton_reduce_sum_int {
  using reduction_type = int;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    int value) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(number_of_1d_array_elements *
                                 (number_of_1d_array_elements - 1) / 2 ==
                             value);
    return std::forward_as_tuple(std::move(box));
  }
};

struct singleton_reduce_sum_double {
  using reduction_type = double;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    double value) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(approx(13.4 * number_of_1d_array_elements) ==
                             value);
    return std::forward_as_tuple(std::move(box));
  }
};

struct singleton_reduce_product_double {
  using reduction_type = double;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    double value) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(approx(pow<number_of_1d_array_elements>(13.4)) ==
                             value);
    return std::forward_as_tuple(std::move(box));
  }
};

struct singleton_reduce_custom_reduction {
  using reduction_type =
      Parallel::ReductionData<int, std::unordered_map<std::string, int>,
                              std::vector<int>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      Parallel::ReductionData<int, std::unordered_map<std::string, int>,
                              std::vector<int>>
          value) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(Parallel::get<0>(value) == 10);
    SPECTRE_PARALLEL_REQUIRE(Parallel::get<1>(value).at("unity") ==
                             number_of_1d_array_elements - 1);
    SPECTRE_PARALLEL_REQUIRE(Parallel::get<1>(value).at("double") ==
                             2 * number_of_1d_array_elements - 2);
    SPECTRE_PARALLEL_REQUIRE(Parallel::get<1>(value).at("negative") == 0);
    SPECTRE_PARALLEL_REQUIRE(
        Parallel::get<2>(value) ==
        (std::vector<int>{
            number_of_1d_array_elements * (number_of_1d_array_elements - 1) / 2,
            number_of_1d_array_elements * 10,
            -8 * number_of_1d_array_elements}));
    return std::forward_as_tuple(std::move(box));
  }
};

template <class Metavariables>
struct SingletonParallelComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = typelist<>;
  using options = typelist<>;
  using metavariables = Metavariables;
  using action_list = typelist<>;
  using initial_databox = db::DataBox<db::get_databox_list<typelist<>>>;
  using reduction_actions_list =
      tmpl::list<singleton_reduce_sum_int, singleton_reduce_sum_double,
                 singleton_reduce_product_double,
                 singleton_reduce_custom_reduction>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    if (next_phase == Metavariables::Phase::CallArrayReduce) {
      auto& local_cache = *(global_cache.ckLocalBranch());
      return;
    }
  }
};

template <class Metavariables>
struct ArrayParallelComponent;

struct array_reduce {
  using apply_args = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* /*meta*/) noexcept {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    int my_send_int = array_index;
    /// [contribute_to_reduction_example]
    Parallel::contribute_to_reduction<ArrayParallelComponent<Metavariables>,
                                      SingletonParallelComponent<Metavariables>,
                                      singleton_reduce_sum_int>(
        cache, array_index, CkReduction::sum_int, my_send_int);
    /// [contribute_to_reduction_example]

    double my_send_double = 13.4;
    Parallel::contribute_to_reduction<ArrayParallelComponent<Metavariables>,
                                      SingletonParallelComponent<Metavariables>,
                                      singleton_reduce_sum_double>(
        cache, array_index, CkReduction::sum_double, my_send_double);

    Parallel::contribute_to_reduction<ArrayParallelComponent<Metavariables>,
                                      SingletonParallelComponent<Metavariables>,
                                      singleton_reduce_product_double>(
        cache, array_index, CkReduction::product_double, my_send_double);

    std::unordered_map<std::string, int> my_send_map;
    my_send_map["unity"] = array_index;
    my_send_map["double"] = 2 * array_index;
    my_send_map["negative"] = -array_index;
    /// [custom_contribute_to_reduction_example]
    Parallel::contribute_to_reduction<ArrayParallelComponent<Metavariables>,
                                      SingletonParallelComponent<Metavariables>,
                                      singleton_reduce_custom_reduction>(
        cache, array_index, reduction_data_reducer,
        Parallel::ReductionData<int, std::unordered_map<std::string, int>,
                                std::vector<int>>{
            10, std::move(my_send_map), std::vector<int>{array_index, 10, -8}});
    /// [custom_contribute_to_reduction_example]

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
  using initial_databox = db::DataBox<db::get_databox_list<typelist<>>>;

  using explicit_single_actions_list = tmpl::list<array_reduce>;

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
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::CallArrayReduce) {
      Parallel::get_parallel_component<ArrayParallelComponent>(local_cache)
          .template explicit_single_action<array_reduce>();
    }
  }
};

struct TestMetavariables {
  using component_list =
      tmpl::list<SingletonParallelComponent<TestMetavariables>,
                 ArrayParallelComponent<TestMetavariables>>;

  static constexpr const char* const help{"Test reductions using Algorithm"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  enum class Phase { Initialization, CallArrayReduce, Exit };
  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        TestMetavariables>& /*cache_proxy*/) {
    return current_phase == Phase::Initialization ? Phase::CallArrayReduce
                                                  : Phase::Exit;
  }
};

/// [custom_reduce_register]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &register_custom_reductions};
/// [custom_reduce_register]
static const std::vector<void (*)()> charm_init_proc_funcs{};

using charm_metavariables = TestMetavariables;

#include "Parallel/CharmMain.cpp"
