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
#include "ErrorHandling/Error.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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
/// [mutate_global_cache_item_mutator]
namespace MutationFunctions {
struct add_stored_double {
  static void apply(const gsl::not_null<std::vector<double>*> data,
                    const double new_value) noexcept {
    data->emplace_back(new_value);
  }
};
}  // namespace MutationFunctions
/// [mutate_global_cache_item_mutator]

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
    /// [mutate_global_cache_item]
    Parallel::mutate<Tags::VectorOfDoubles,
                     MutationFunctions::add_stored_double>(cache.thisProxy,
                                                           42.0);
    /// [mutate_global_cache_item]
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
    auto callback = CkCallback(
        Parallel::index_from_parallel_component<ParallelComponent>::
            template simple_action<simple_action_check_and_use_stored_double>(),
        this_proxy);
    const bool is_ready =
        ::Parallel::mutable_cache_item_is_ready<Tags::VectorOfDoubles>(
            cache,
            [&callback](const std::vector<double>& VectorOfDoubles)
                -> std::optional<CkCallback> {
              return VectorOfDoubles.empty()
                         ? std::optional<CkCallback>(callback)
                         : std::optional<CkCallback>{};
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
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    ++number_of_calls_to_use_stored_double_apply;
    const std::vector<double> expected_result{42.0};
    /// [retrieve_mutable_cache_item]
    SPECTRE_PARALLEL_REQUIRE(Parallel::get<Tags::VectorOfDoubles>(cache) ==
                             expected_result);
    /// [retrieve_mutable_cache_item]
    return std::tuple<db::DataBox<DbTags>&&, bool>(std::move(box), true);
  }

  /// [check_mutable_cache_item_is_ready]
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& /*box*/,
                       const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                       Parallel::GlobalCache<Metavariables>& cache,
                       const ArrayIndex& /*array_index*/) noexcept {
    ++number_of_calls_to_use_stored_double_is_ready;
    auto& this_proxy = Parallel::get_parallel_component<
        UseMutatedCacheComponent<Metavariables>>(cache);
    auto callback = CkCallback(
        Parallel::index_from_parallel_component<
            UseMutatedCacheComponent<Metavariables>>::perform_algorithm(),
        this_proxy);
    return ::Parallel::mutable_cache_item_is_ready<Tags::VectorOfDoubles>(
        cache,
        [&callback](const std::vector<double>& VectorOfDoubles)
            -> std::optional<CkCallback> {
          return VectorOfDoubles.empty() ? std::optional<CkCallback>(callback)
                                         : std::optional<CkCallback>{};
        });
  }
  /// [check_mutable_cache_item_is_ready]
};
}  // namespace Actions

}  // namespace mutate_cache

// We have three ParallelComponents:
//
// 1) MutateCacheComponent mutates the value in the GlobalCache using
//    simple_actions, and then tests that the value in the GlobalCache
//    is correct, using simple_actions.
//
// 2) UseMutatedCacheComponent has a single iterable_action whose
//    is_ready function checks the size of the value in the
//    GlobalCache.  If the size is correct, then its apply function
//    verifies that the value is correct.
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
    auto& local_cache = *(global_cache.ckLocalBranch());
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
    auto& local_cache = *(global_cache.ckLocalBranch());
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
    auto& local_cache = *(global_cache.ckLocalBranch());
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

  static Phase determine_next_phase(
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
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
