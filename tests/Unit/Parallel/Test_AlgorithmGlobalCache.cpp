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
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
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
  static std::string name() noexcept { return "VectorOfDoubles"; }
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
  static type create_from_options(const type& input_type) noexcept {
    return input_type;
  }
};
}  // namespace Tags

// Functions to be passed into GlobalCache::mutate
// [mutate_global_cache_item_mutator]
namespace MutationFunctions {
struct add_stored_double {
  static void apply(const gsl::not_null<std::vector<double>*> data,
                    const double new_value) noexcept {
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
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const  // NOLINT const
                    /*meta*/) noexcept {
    return std::make_tuple(
        db::create_from<db::RemoveTags<>, db::AddSimpleTags<>>(std::move(box)),
        true);
  }
};

struct add_new_stored_double {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) noexcept {
    // [mutate_global_cache_item]
    Parallel::mutate<Tags::VectorOfDoubles,
                     MutationFunctions::add_stored_double>(cache.thisProxy,
                                                           42.0);
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
                    const ArrayIndex& /*array_index*/) noexcept {
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
                    const ArrayIndex& /*array_index*/) noexcept {
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
  static std::tuple<db::DataBox<DbTags>&&, Parallel::AlgorithmExecution> apply(
      db::DataBox<DbTags>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
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
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }
    // [check_mutable_cache_item_is_ready]

    ++number_of_calls_to_use_stored_double_apply;
    const std::vector<double> expected_result{42.0};
    // [retrieve_mutable_cache_item]
    SPECTRE_PARALLEL_REQUIRE(Parallel::get<Tags::VectorOfDoubles>(cache) ==
                             expected_result);
    // [retrieve_mutable_cache_item]
    return {std::move(box), Parallel::AlgorithmExecution::Pause};
  }
};
}  // namespace Actions

}  // namespace mutate_cache

// We have three ParallelComponents:
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
template <class Metavariables>
struct MutateCacheComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using mutable_global_cache_tags =
      tmpl::list<mutate_cache::Tags::VectorOfDoubles>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<mutate_cache::Actions::initialize>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<MutateCacheComponent>(local_cache)
        .start_phase(next_phase);
    if (next_phase == Metavariables::Phase::MutableCacheStart) {
      Parallel::simple_action<mutate_cache::Actions::add_new_stored_double>(
          Parallel::get_parallel_component<MutateCacheComponent>(local_cache));
    } else if (next_phase == Metavariables::Phase::MutableCacheFinish) {
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
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<mutate_cache::Actions::initialize>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::MutableCacheStart,
          tmpl::list<mutate_cache::Actions::use_stored_double>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
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
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<mutate_cache::Actions::initialize>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<CheckAndUseMutatedCacheComponent>(
        local_cache)
        .start_phase(next_phase);
    if (next_phase == Metavariables::Phase::MutableCacheSimpleActionStart) {
      Parallel::simple_action<
          mutate_cache::Actions::simple_action_check_and_use_stored_double>(
          Parallel::get_parallel_component<CheckAndUseMutatedCacheComponent>(
              local_cache));
    }
  }
};

struct TestMetavariables {
  using component_list =
      tmpl::list<MutateCacheComponent<TestMetavariables>,
                 UseMutatedCacheComponent<TestMetavariables>,
                 CheckAndUseMutatedCacheComponent<TestMetavariables>>;

  static constexpr Options::String help =
      "An executable for testing mutable items in the GlobalCache.";

  enum class Phase {
    Initialization,
    MutableCacheSimpleActionStart,
    MutableCacheStart,
    MutableCacheFinish,
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
        return Phase::MutableCacheSimpleActionStart;
      case Phase::MutableCacheSimpleActionStart:
        return Phase::MutableCacheStart;
      case Phase::MutableCacheStart:
        return Phase::MutableCacheFinish;
      case Phase::MutableCacheFinish:
        [[fallthrough]];
      case Phase::Exit:
        return Phase::Exit;
      default:
        ERROR("Unknown Phase...");
    }

    return Phase::Exit;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
