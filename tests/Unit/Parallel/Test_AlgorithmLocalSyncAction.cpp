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
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
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

namespace LocalSyncActionTest {
template <class Metavariables>
struct NodegroupComponent;

struct StepNumber : db::SimpleTag {
  using type = size_t;
};

struct InitializeNodegroup {
  using simple_tags = tmpl::list<StepNumber>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // default initialization of SimpleTag is fine
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

// [synchronous_action_example]
struct SyncGetPointerFromNodegroup {
  using return_type = size_t*;
  template <typename ParallelComponent, typename DbTagsList>
  static size_t* apply(db::DataBox<DbTagsList>& box,
                       const gsl::not_null<Parallel::NodeLock*> node_lock) {
    if constexpr (tmpl::list_contains_v<DbTagsList, StepNumber>) {
      size_t* result = nullptr;
      // We must lock access to the box, because box access is non-atomic and
      // nodegroups can have multiple actions running in separate threads. Once
      // we retrieve the pointer to data from the box, we can safely pass it
      // along because we know that there are no compute tags that depend on
      // `StepNumber`. Without that guarantee, this would not be a supported use
      // of the box.
      node_lock->lock();
      db::mutate<StepNumber>(
          [&result](const gsl::not_null<size_t*> step_number) {
            result = step_number;
          },
          make_not_null(&box));
      node_lock->unlock();
      return result;
    } else {
      // avoid 'unused' warnings
      (void)node_lock;
      ERROR("Could not find required tag `StepNumber` in the databox");
    }
  }
};
// [synchronous_action_example]

struct SyncGetConstRefFromNodegroup {
  using return_type = const size_t&;
  template <typename ParallelComponent, typename DbTagsList>
  static const size_t& apply(
      db::DataBox<DbTagsList>& box,
      const gsl::not_null<Parallel::NodeLock*> node_lock) {
    if constexpr (tmpl::list_contains_v<DbTagsList, StepNumber>) {
      // We must lock access to the box, because box access is non-atomic and
      // nodegroups can have multiple actions running in separate threads. Once
      // we retrieve the pointer to data from the box, we can safely pass it
      // along because we know that there are no compute tags that depend on
      // `StepNumber`. Without that guarantee, this would not be a supported use
      // of the box.
      node_lock->lock();
      const size_t& result = db::get<StepNumber>(box);
      node_lock->unlock();
      return result;
    } else {
      // avoid 'unused' warnings
      (void)node_lock;
      ERROR("Could not find required tag `StepNumber` in the databox");
    }
  }
};

struct IncrementNodegroupStep {
  using return_type = void;
  template <typename ParallelComponent, typename DbTagsList>
  static void apply(db::DataBox<DbTagsList>& box,
                    const gsl::not_null<Parallel::NodeLock*> node_lock) {
    if constexpr (tmpl::list_contains_v<DbTagsList, StepNumber>) {
      // We must lock access to the box, because box access is non-atomic and
      // nodegroups can have multiple actions running in separate threads.
      node_lock->lock();
      db::mutate<StepNumber>(
          [](const gsl::not_null<size_t*> step_number) { ++(*step_number); },
          make_not_null(&box));
      node_lock->unlock();
    } else {
      // avoid 'unused' warnings
      (void)node_lock;
      ERROR("Could not find required tag `StepNumber` in the databox");
    }
  }
};

struct TestSyncActionIncrement {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // [synchronous_action_invocation_example]
    size_t* step_number =
        Parallel::local_synchronous_action<SyncGetPointerFromNodegroup>(
            Parallel::get_parallel_component<NodegroupComponent<Metavariables>>(
                cache));
    // [synchronous_action_invocation_example]
    Parallel::local_synchronous_action<IncrementNodegroupStep>(
        Parallel::get_parallel_component<NodegroupComponent<Metavariables>>(
            cache));
    SPECTRE_PARALLEL_REQUIRE(*step_number == 1);
    ++(*step_number);
    Parallel::local_synchronous_action<IncrementNodegroupStep>(
        Parallel::get_parallel_component<NodegroupComponent<Metavariables>>(
            cache));
    SPECTRE_PARALLEL_REQUIRE(*step_number == 3);
    SPECTRE_PARALLEL_REQUIRE(
        *step_number ==
        Parallel::local_synchronous_action<SyncGetConstRefFromNodegroup>(
            Parallel::get_parallel_component<NodegroupComponent<Metavariables>>(
                cache)));
    SPECTRE_PARALLEL_REQUIRE(
        step_number ==
        &(Parallel::local_synchronous_action<SyncGetConstRefFromNodegroup>(
            Parallel::get_parallel_component<NodegroupComponent<Metavariables>>(
                cache))));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <class Metavariables>
struct NodegroupComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        tmpl::list<InitializeNodegroup>>>;
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
struct ArrayComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Evolve,
      tmpl::list<TestSyncActionIncrement, Parallel::Actions::TerminatePhase>>>;
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
}  // namespace LocalSyncActionTest

struct TestMetavariables {
  using component_list =
      tmpl::list<LocalSyncActionTest::ArrayComponent<TestMetavariables>,
                 LocalSyncActionTest::NodegroupComponent<TestMetavariables>>;

  static constexpr Options::String help = "";

  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Evolve,
       Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
