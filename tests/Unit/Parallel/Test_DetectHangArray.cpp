// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace DetectHang {
struct CheckNextIterableAction {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    if constexpr (Parallel::is_array_v<ParallelComponent>) {
      const auto* local_object =
          Parallel::local(Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index]);
      SPECTRE_PARALLEL_REQUIRE(local_object != nullptr);
      SPECTRE_PARALLEL_REQUIRE(
          local_object->deadlock_analysis_next_iterable_action() ==
          std::string("Hang"));
    } else if constexpr (Parallel::is_singleton_v<ParallelComponent>) {
      const auto* local_object = Parallel::local(
          Parallel::get_parallel_component<ParallelComponent>(cache));
      SPECTRE_PARALLEL_REQUIRE(local_object != nullptr);
      SPECTRE_PARALLEL_REQUIRE(
          local_object->deadlock_analysis_next_iterable_action() ==
          std::string("Hang"));
    } else {
      const auto* local_object = Parallel::local_branch(
          Parallel::get_parallel_component<ParallelComponent>(cache));
      SPECTRE_PARALLEL_REQUIRE(local_object != nullptr);
      SPECTRE_PARALLEL_REQUIRE(
          local_object->deadlock_analysis_next_iterable_action() ==
          std::string("Hang"));
    }
    Parallel::printf("Succeeded for %s\n",
                     pretty_type::name<ParallelComponent>());
  }
};

struct Hang {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Retry, std::nullopt};
  }
};

template <class Metavariables>
struct NodegroupComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<Hang>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<NodegroupComponent>(local_cache)
        .start_phase(next_phase);
  }
};

template <class Metavariables>
struct GroupComponent {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<Hang>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<GroupComponent>(local_cache)
        .start_phase(next_phase);
  }
};

template <class Metavariables>
struct SingletonComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<Hang>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<SingletonComponent>(local_cache)
        .start_phase(next_phase);
  }
};

template <class Metavariables>
struct ArrayComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Evolve, tmpl::list<Hang>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const std::unordered_set<size_t>& /*procs_to_ignore*/ = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayComponent>(local_cache);
    // we only want one array component for this test.
    array_proxy[0].insert(global_cache, tuples::TaggedTuple<>{}, 0);
    array_proxy.doneInserting();
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<ArrayComponent>(local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace DetectHang

struct TestMetavariables {
  using component_list =
      tmpl::list<DetectHang::ArrayComponent<TestMetavariables>,
                 DetectHang::SingletonComponent<TestMetavariables>,
                 DetectHang::GroupComponent<TestMetavariables>,
                 DetectHang::NodegroupComponent<TestMetavariables>>;

  static constexpr Options::String help = "";

  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Evolve,
       Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  static void run_deadlock_analysis_simple_actions(
      Parallel::GlobalCache<TestMetavariables>& cache,
      const std::vector<std::string>& deadlocked_components) {
    SPECTRE_PARALLEL_REQUIRE(
        alg::find(deadlocked_components, std::string{"ArrayComponent"}) !=
        deadlocked_components.end());
    SPECTRE_PARALLEL_REQUIRE(
        alg::find(deadlocked_components, std::string{"SingletonComponent"}) !=
        deadlocked_components.end());
    SPECTRE_PARALLEL_REQUIRE(
        alg::find(deadlocked_components, std::string{"GroupComponent"}) !=
        deadlocked_components.end());
    SPECTRE_PARALLEL_REQUIRE(
        alg::find(deadlocked_components, std::string{"NodegroupComponent"}) !=
        deadlocked_components.end());

    Parallel::simple_action<DetectHang::CheckNextIterableAction>(
        Parallel::get_parallel_component<
            DetectHang::ArrayComponent<TestMetavariables>>(cache));
    Parallel::simple_action<DetectHang::CheckNextIterableAction>(
        Parallel::get_parallel_component<
            DetectHang::SingletonComponent<TestMetavariables>>(cache));
    Parallel::simple_action<DetectHang::CheckNextIterableAction>(
        Parallel::get_parallel_component<
            DetectHang::GroupComponent<TestMetavariables>>(cache));
    Parallel::simple_action<DetectHang::CheckNextIterableAction>(
        Parallel::get_parallel_component<
            DetectHang::NodegroupComponent<TestMetavariables>>(cache));
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
