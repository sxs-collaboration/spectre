// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AlgorithmArray.hpp"
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
struct TestMetavariables;
template <class Metavariables>
struct SingletonParallelComponent;

// The reason we use a 46 element array is that on Wheeler, the SXS
// supercomputer at Caltech, there are 23 worker threads per node and we want to
// be able to test on two nodes to make sure multinode communication is working
// correctly.
static constexpr int number_of_1d_array_elements = 46;

/// [custom_reduce_function]
CkReductionMsg* reduce_reduction_data(const int number_of_messages,
                                      CkReductionMsg** const msgs) noexcept {
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

/// [reduce_sum_int_action]
struct ProcessReducedSumOfInts {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const int& value) noexcept {
    SPECTRE_PARALLEL_REQUIRE(number_of_1d_array_elements *
                                 (number_of_1d_array_elements - 1) / 2 ==
                             value);
  }
};
/// [reduce_sum_int_action]

struct ProcessReducedSumOfDoubles {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double& value) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(approx(13.4 * number_of_1d_array_elements) ==
                             value);
  }
};

struct ProcessReducedProductOfDoubles {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double& value) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent,
                         SingletonParallelComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(approx(pow<number_of_1d_array_elements>(13.4)) ==
                             value);
  }
};

/// [custom_reduction_action]
struct ProcessCustomReductionAction {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(
      db::DataBox<DbTags>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const Parallel::ReductionData<int, std::unordered_map<std::string, int>,
                                    std::vector<int>>&
          value) noexcept {
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
  }
};
/// [custom_reduction_action]

template <class Metavariables>
struct SingletonParallelComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<tmpl::list<>>;

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

struct ArrayReduce {
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
    const auto& my_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent<Metavariables>>(
            cache)[array_index];
    const auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent<Metavariables>>(
            cache);
    const auto& singleton_proxy = Parallel::get_parallel_component<
        SingletonParallelComponent<Metavariables>>(cache);
    /// [contribute_to_reduction_example]
    Parallel::contribute_to_reduction<ProcessReducedSumOfInts>(
        my_send_int, my_proxy, singleton_proxy, CkReduction::sum_int);
    /// [contribute_to_reduction_example]
    /// [contribute_to_broadcast_reduction]
    Parallel::contribute_to_reduction<ProcessReducedSumOfInts>(
        my_send_int, my_proxy, array_proxy, CkReduction::sum_int);
    /// [contribute_to_broadcast_reduction]

    double my_send_double = 13.4;
    Parallel::contribute_to_reduction<ProcessReducedSumOfDoubles>(
        my_send_double, my_proxy, singleton_proxy, CkReduction::sum_double);

    Parallel::contribute_to_reduction<ProcessReducedProductOfDoubles>(
        my_send_double, my_proxy, singleton_proxy, CkReduction::product_double);

    /// [custom_contribute_to_reduction_example]
    std::unordered_map<std::string, int> my_send_map;
    my_send_map["unity"] = array_index;
    my_send_map["double"] = 2 * array_index;
    my_send_map["negative"] = -array_index;
    Parallel::contribute_to_reduction<&reduce_reduction_data,
                                      ProcessCustomReductionAction>(
        Parallel::ReductionData<int, std::unordered_map<std::string, int>,
                                std::vector<int>>{
            10, my_send_map, std::vector<int>{array_index, 10, -8}},
        my_proxy, singleton_proxy);
    /// [custom_contribute_to_reduction_example]
    /// [custom_contribute_to_broadcast_reduction]
    Parallel::contribute_to_reduction<&reduce_reduction_data,
                                      ProcessCustomReductionAction>(
        Parallel::ReductionData<int, std::unordered_map<std::string, int>,
                                std::vector<int>>{
            10, my_send_map, std::vector<int>{array_index, 10, -8}},
        my_proxy, array_proxy);
    /// [custom_contribute_to_broadcast_reduction]

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
      Parallel::simple_action<ArrayReduce>(
          Parallel::get_parallel_component<ArrayParallelComponent>(
              local_cache));
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

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.cpp"
