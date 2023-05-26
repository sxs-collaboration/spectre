// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Matrix.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Actions/GetLockPointer.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ContributeMemoryData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PUP {
class er;
}  // namespace PUP

template <class Metavariables>
struct MutateCacheComponent;
template <class Metavariables>
struct UseMutatedCacheComponent;

namespace mutate_cache {

// An option tag is needed for every type the GlobalCache contains.
namespace OptionTags {
struct VectorOfDoubles {
  static std::string name() { return "VectorOfDoubles"; }
  static constexpr Options::String help = "Options for vector of doubles";
  using type = std::vector<double>;
};
}  // namespace OptionTags

// Tag to label the quantity in the GlobalCache.
namespace Tags {
struct VectorOfDoubles : db::SimpleTag {
  using type = typename OptionTags::VectorOfDoubles::type;
  using option_tags = tmpl::list<OptionTags::VectorOfDoubles>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& input_type) { return input_type; }
};
}  // namespace Tags

// Functions to be passed into GlobalCache::mutate
// [mutate_global_cache_item_mutator]
namespace MutationFunctions {
struct add_stored_double {
  static void apply(const gsl::not_null<std::vector<double>*> data,
                    const double new_value) {
    data->emplace_back(new_value);
  }
};
}  // namespace MutationFunctions
// [mutate_global_cache_item_mutator]

namespace Actions {
struct initialize {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const  // NOLINT const
      /*meta*/) {
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct add_new_stored_double {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    // [mutate_global_cache_item]
    Parallel::mutate<Tags::VectorOfDoubles,
                     MutationFunctions::add_stored_double>(cache, 42.0);
    // [mutate_global_cache_item]
  }
};

// Global variables to make sure that certain functions have been called.
size_t number_of_calls_to_use_stored_double_is_ready = 0;
size_t number_of_calls_to_use_stored_double_apply = 0;
size_t number_of_calls_to_check_and_use_stored_double_is_ready = 0;
size_t number_of_calls_to_check_and_use_stored_double_apply = 0;

struct finalize {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    const std::vector<double> expected_result{42.0};
    SPECTRE_PARALLEL_REQUIRE(Parallel::get<Tags::VectorOfDoubles>(cache) ==
                             expected_result);
    SPECTRE_PARALLEL_REQUIRE(number_of_calls_to_use_stored_double_is_ready ==
                             2);
    SPECTRE_PARALLEL_REQUIRE(number_of_calls_to_use_stored_double_apply == 1);
    SPECTRE_PARALLEL_REQUIRE(
        number_of_calls_to_check_and_use_stored_double_is_ready == 2);
    SPECTRE_PARALLEL_REQUIRE(
        number_of_calls_to_check_and_use_stored_double_apply == 1);
  }
};

struct simple_action_check_and_use_stored_double {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    ++number_of_calls_to_check_and_use_stored_double_is_ready;
    auto& this_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const bool is_ready =
        ::Parallel::mutable_cache_item_is_ready<Tags::VectorOfDoubles>(
            cache,
            [&this_proxy](const std::vector<double>& VectorOfDoubles)
                -> std::unique_ptr<Parallel::Callback> {
              return VectorOfDoubles.empty()
                         ? std::unique_ptr<Parallel::Callback>(
                               new Parallel::SimpleActionCallback<
                                   simple_action_check_and_use_stored_double,
                                   decltype(this_proxy)>(this_proxy))
                         : std::unique_ptr<Parallel::Callback>{};
            });

    if (is_ready) {
      ++number_of_calls_to_check_and_use_stored_double_apply;
      const std::vector<double> expected_result{42.0};
      SPECTRE_PARALLEL_REQUIRE(Parallel::get<Tags::VectorOfDoubles>(cache) ==
                               expected_result);
    }
  }
};

struct use_stored_double {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& /*box*/,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    ++number_of_calls_to_use_stored_double_is_ready;
    // [check_mutable_cache_item_is_ready]
    auto& this_proxy = Parallel::get_parallel_component<
        UseMutatedCacheComponent<Metavariables>>(cache);
    if (not ::Parallel::mutable_cache_item_is_ready<Tags::VectorOfDoubles>(
            cache,
            [&this_proxy](const std::vector<double>& VectorOfDoubles)
                -> std::unique_ptr<Parallel::Callback> {
              return VectorOfDoubles.empty()
                         ? std::unique_ptr<Parallel::Callback>(
                               new Parallel::PerformAlgorithmCallback(
                                   this_proxy))
                         : std::unique_ptr<Parallel::Callback>{};
            })) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    // [check_mutable_cache_item_is_ready]

    ++number_of_calls_to_use_stored_double_apply;
    const std::vector<double> expected_result{42.0};
    // [retrieve_mutable_cache_item]
    SPECTRE_PARALLEL_REQUIRE(Parallel::get<Tags::VectorOfDoubles>(cache) ==
                             expected_result);
    // [retrieve_mutable_cache_item]
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};
}  // namespace Actions

}  // namespace mutate_cache

// We have five ParallelComponents:
//
// 1) MutateCacheComponent mutates the value in the GlobalCache using
//    simple_actions, and then tests that the value in the GlobalCache
//    is correct, using simple_actions.
//
// 2) UseMutatedCacheComponent has a single iterable_action that waits
//    for the size of the value in the GlobalCache to be correct.
//
// 3) CheckAndUseMutatedCacheComponent has a simple_action that checks
//    the size of the value in the GlobalCache, and then if the size
//    is correct, it verifies that its value is correct.
//
// 4) CheckParallelInfo checks the parallel info functions of the GlobalCache
//    against the Parallel:: and sys:: functions.
//
// 5) CheckMemoryMonitorRelatedMethods calls the
//    `compute_size_for_memory_monitor` entry method of the GlobalCache and the
//    MutableGlobalCache which requires there to be a MemoryMonitor in the
//    component list (and also an ObserverWriter for writing data)
template <class Metavariables>
struct MutateCacheComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using mutable_global_cache_tags =
      tmpl::list<mutate_cache::Tags::VectorOfDoubles>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<mutate_cache::Actions::initialize>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<MutateCacheComponent>(local_cache)
        .start_phase(next_phase);
    if (next_phase == Parallel::Phase::Solve) {
      Parallel::simple_action<mutate_cache::Actions::add_new_stored_double>(
          Parallel::get_parallel_component<MutateCacheComponent>(local_cache));
    } else if (next_phase == Parallel::Phase::Evolve) {
      Parallel::simple_action<mutate_cache::Actions::finalize>(
          Parallel::get_parallel_component<MutateCacheComponent>(local_cache));
    }
  }
};

template <class Metavariables>
struct UseMutatedCacheComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using mutable_global_cache_tags =
      tmpl::list<mutate_cache::Tags::VectorOfDoubles>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<mutate_cache::Actions::initialize>>,
      Parallel::PhaseActions<
          Parallel::Phase::Solve,
          tmpl::list<mutate_cache::Actions::use_stored_double>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<UseMutatedCacheComponent>(local_cache)
        .start_phase(next_phase);
  }
};

template <class Metavariables>
struct CheckAndUseMutatedCacheComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using mutable_global_cache_tags =
      tmpl::list<mutate_cache::Tags::VectorOfDoubles>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<mutate_cache::Actions::initialize>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<CheckAndUseMutatedCacheComponent>(
        local_cache)
        .start_phase(next_phase);
    if (next_phase == Parallel::Phase::Register) {
      Parallel::simple_action<
          mutate_cache::Actions::simple_action_check_and_use_stored_double>(
          Parallel::get_parallel_component<CheckAndUseMutatedCacheComponent>(
              local_cache));
    }
  }
};

template <class Metavariables>
struct CheckParallelInfo {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<CheckParallelInfo>(cache).start_phase(
        next_phase);
    if (next_phase == Parallel::Phase::Execute) {
      // Check parallel info
      SPECTRE_PARALLEL_REQUIRE(cache.number_of_procs() ==
                               sys::number_of_procs());
      SPECTRE_PARALLEL_REQUIRE(cache.number_of_nodes() ==
                               sys::number_of_nodes());
      SPECTRE_PARALLEL_REQUIRE(cache.procs_on_node(0) == sys::procs_on_node(0));
      SPECTRE_PARALLEL_REQUIRE(cache.first_proc_on_node(0) ==
                               sys::first_proc_on_node(0));
      SPECTRE_PARALLEL_REQUIRE(cache.node_of(0) == sys::node_of(0));
      SPECTRE_PARALLEL_REQUIRE(cache.local_rank_of(0) == sys::local_rank_of(0));
      SPECTRE_PARALLEL_REQUIRE(cache.my_proc() == sys::my_proc());
      SPECTRE_PARALLEL_REQUIRE(cache.my_node() == sys::my_node());
      SPECTRE_PARALLEL_REQUIRE(cache.my_local_rank() == sys::my_local_rank());
      SPECTRE_PARALLEL_REQUIRE(Parallel::number_of_procs<int>(cache) ==
                               sys::number_of_procs());
      SPECTRE_PARALLEL_REQUIRE(Parallel::number_of_nodes<int>(cache) ==
                               sys::number_of_nodes());
      SPECTRE_PARALLEL_REQUIRE(Parallel::procs_on_node<int>(0, cache) ==
                               sys::procs_on_node(0));
      SPECTRE_PARALLEL_REQUIRE(Parallel::first_proc_on_node<int>(0, cache) ==
                               sys::first_proc_on_node(0));
      SPECTRE_PARALLEL_REQUIRE(Parallel::node_of<int>(0, cache) ==
                               sys::node_of(0));
      SPECTRE_PARALLEL_REQUIRE(Parallel::local_rank_of<int>(0, cache) ==
                               sys::local_rank_of(0));
      SPECTRE_PARALLEL_REQUIRE(Parallel::my_proc<int>(cache) == sys::my_proc());
      SPECTRE_PARALLEL_REQUIRE(Parallel::my_node<int>(cache) == sys::my_node());
      SPECTRE_PARALLEL_REQUIRE(Parallel::my_local_rank<int>(cache) ==
                               sys::my_local_rank());
    }
  }
};

template <class Metavariables>
struct CheckMemoryMonitorRelatedMethods {
 private:
  static inline double cache_size_ = 0.0;
  static inline double mutable_cache_size_ = 0.0;
  static inline std::string filename_{"Test_AlgorithmGlobalCacheReduction.h5"};

 public:
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<CheckMemoryMonitorRelatedMethods>(
        local_cache)
        .start_phase(next_phase);

    const double time = 1.0;
    if (next_phase == Parallel::Phase::Testing) {
      // Remove the file from a previous test run if it exists
      if (file_system::check_if_file_exists(filename_)) {
        file_system::rm(filename_, true);
      }
      // This will broadcast to all branches of the global cache. This
      // executable is run on 1 core so there is only 1 branch. The time is
      // arbitrary
      local_cache.compute_size_for_memory_monitor(time);

      // This will broadcast to all branches of the mutable global cache. This
      // executable is run on 1 core so there is only 1 branch. The time is
      // arbitrary
      auto mutable_global_cache_proxy =
          local_cache.mutable_global_cache_proxy();
      mutable_global_cache_proxy.compute_size_for_memory_monitor(global_cache,
                                                                 time);

      // Store the values for testing. The only reason this works is because we
      // are running on 1 core, so we are always on the "local" branch of an
      // object whether that object be a group or nodegroup. If this is run on
      // more than one core, this will still run, but the values will be
      // incorrect for the test
      SPECTRE_PARALLEL_REQUIRE(Parallel::number_of_procs<int>(local_cache) ==
                               1);
      cache_size_ = size_of_object_in_bytes(local_cache) / 1.0e6;
      mutable_cache_size_ = size_of_object_in_bytes(*Parallel::local_branch(
                                local_cache.mutable_global_cache_proxy())) /
                            1.0e6;
    } else if (next_phase == Parallel::Phase::Cleanup) {
      auto hdf5_lock =
          Parallel::local_branch(
              Parallel::get_parallel_component<
                  observers::ObserverWriter<Metavariables>>(local_cache))
              ->template local_synchronous_action<
                  observers::Actions::GetLockPointer<
                      observers::Tags::H5FileLock>>();

      hdf5_lock->lock();
      SPECTRE_PARALLEL_REQUIRE(file_system::check_if_file_exists(filename_));

      const h5::H5File<h5::AccessType::ReadOnly> read_file{filename_};

      const std::vector<std::string> cache_legend{
          {"Time", "Size on node 0 (MB)", "Average size per node (MB)"}};
      const std::vector<std::string> mutable_cache_legend{
          {"Time", "Size on node 0 (MB)", "Proc of max size",
           "Size on proc of max size (MB)", "Average size per node (MB)"}};

      const std::string cache_name{"/MemoryMonitors/GlobalCache"};
      const std::string mutable_cache_name{
          "/MemoryMonitors/MutableGlobalCache"};

      const auto check_caches = [&read_file, &time](
                                    const std::string& name,
                                    const std::vector<std::string>& legend,
                                    const double check_size) {
        const auto& dataset = read_file.get<h5::Dat>(name, legend);
        const Matrix data = dataset.get_data();
        const auto& legend_file = dataset.get_legend();

        SPECTRE_PARALLEL_REQUIRE(legend == legend_file);
        SPECTRE_PARALLEL_REQUIRE(data.columns() == legend.size());
        SPECTRE_PARALLEL_REQUIRE(data.rows() == 1);
        // First column is always time
        SPECTRE_PARALLEL_REQUIRE(data(0, 0) == time);
        // Second column is size on node 0 (since we are running on only one
        // node)
        SPECTRE_PARALLEL_REQUIRE(data(0, 1) == check_size);
        // Last column is always average per node, but since we are running on
        // one node, this should be the same as the size on node 0
        SPECTRE_PARALLEL_REQUIRE(data(0, data.columns() - 1) == check_size);
        // For groups, check the max proc and size on that proc. Since we are
        // running on one proc, this will always be proc 0 and the size will be
        // the same as that on node 0
        if (data.columns() > 3) {
          SPECTRE_PARALLEL_REQUIRE(data(0, 2) == 0.0);
          SPECTRE_PARALLEL_REQUIRE(data(0, 3) == check_size);
        }

        read_file.close_current_object();
      };

      check_caches(cache_name, cache_legend, cache_size_);
      check_caches(mutable_cache_name, mutable_cache_legend,
                   mutable_cache_size_);

      hdf5_lock->unlock();

      // Remove the file once we're done
      if (file_system::check_if_file_exists(filename_)) {
        file_system::rm(filename_, true);
      }
    }
  }
};

struct TestMetavariables {
  using observed_reduction_data_tags = tmpl::list<>;
  using component_list =
      tmpl::list<MutateCacheComponent<TestMetavariables>,
                 UseMutatedCacheComponent<TestMetavariables>,
                 CheckAndUseMutatedCacheComponent<TestMetavariables>,
                 CheckParallelInfo<TestMetavariables>,
                 mem_monitor::MemoryMonitor<TestMetavariables>,
                 CheckMemoryMonitorRelatedMethods<TestMetavariables>,
                 observers::ObserverWriter<TestMetavariables>>;

  static constexpr Options::String help =
      "An executable for testing mutable items in the GlobalCache.";

  static constexpr std::array<Parallel::Phase, 18> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::Solve, Parallel::Phase::Evolve,
       Parallel::Phase::Execute, Parallel::Phase::Testing,
       Parallel::Phase::Cleanup, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
