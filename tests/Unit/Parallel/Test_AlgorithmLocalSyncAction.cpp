// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace LocalSyncActionTest {
template <class Metavariables>
struct NodegroupComponent;

struct StepNumber : db::SimpleTag {
  using type = size_t;
};

struct InitializeNodegroup {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (not tmpl::list_contains_v<DbTagsList, StepNumber>) {
      return std::make_tuple(
          db::create_from<db::RemoveTags<>, db::AddSimpleTags<StepNumber>>(
              std::move(box), 0_st),
          true);
    } else {
      return std::make_tuple(std::move(box), true);
    }
  }
};

/// [synchronous_action_example]
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
          make_not_null(&box),
          [&result](const gsl::not_null<size_t*> step_number) noexcept {
            result = step_number;
          });
      node_lock->unlock();
      return result;
    } else {
      // avoid 'unused' warnings
      (void)node_lock;
      ERROR("Could not find required tag `StepNumber` in the databox");
    }
  }
};
/// [synchronous_action_example]

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
          make_not_null(&box),
          [](const gsl::not_null<size_t*> step_number) noexcept {
            ++(*step_number);
          });
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
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    /// [synchronous_action_invocation_example]
    size_t* step_number =
        Parallel::local_synchronous_action<SyncGetPointerFromNodegroup>(
            Parallel::get_parallel_component<NodegroupComponent<Metavariables>>(
                cache));
    /// [synchronous_action_invocation_example]
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

    return std::forward_as_tuple(std::move(box));
  }
};

template <class Metavariables>
struct NodegroupComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<InitializeNodegroup>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
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
      typename Metavariables::Phase, Metavariables::Phase::Evolve,
      tmpl::list<TestSyncActionIncrement, Parallel::Actions::TerminatePhase>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
      /*initialization_items*/) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayComponent>(local_cache);
    // we only want one array component for this test.
    array_proxy[0].insert(global_cache, {}, 0);
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
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

  enum class Phase {
    Initialization,
    Evolve,
    Exit
  };

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<
          tuples::TaggedTuple<Tags...>*> /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    if(current_phase == Phase::Initialization) {
      return Phase::Evolve;
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
