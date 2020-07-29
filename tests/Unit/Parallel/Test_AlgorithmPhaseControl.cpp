// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

namespace Actions {
struct InitializePhaseRecord;
struct RecordCurrentPhase;
template <size_t phase>
struct RecordPhaseIteration;
template <typename ComponentToRestart>
struct RestartMe;
template <typename OtherComponent, size_t interval>
struct TerminateAndRestart;
struct RequestPhaseEntryFunction;
struct RequestPhaseActionReturn;
struct StopForSync;
struct Finalize;
}  // namespace Actions

template <typename Metavariables>
struct ComponentAlpha;

template <typename Metavariables>
struct ComponentBeta {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::InitializePhaseRecord,
                                        Actions::RecordPhaseIteration<0_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::SyncPhaseA,
                             tmpl::list<Actions::RecordPhaseIteration<1_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::SyncPhaseB,
                             tmpl::list<Actions::RecordPhaseIteration<2_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<
              Actions::RecordPhaseIteration<3_st>,
              Actions::TerminateAndRestart<ComponentAlpha<Metavariables>, 3_st>,
              Actions::RequestPhaseActionReturn, Actions::StopForSync>>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<Actions::RecordCurrentPhase>(
        Parallel::get_parallel_component<ComponentBeta>(local_cache),
        next_phase);
    if (next_phase == Metavariables::Phase::Finalize) {
      Parallel::simple_action<Actions::Finalize>(
          Parallel::get_parallel_component<ComponentBeta>(local_cache));
    } else {
      Parallel::get_parallel_component<ComponentBeta>(local_cache)
          .start_phase(next_phase);
    }
  }
};

template <typename Metavariables>
struct ComponentAlpha {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::InitializePhaseRecord,
                                        Actions::RecordPhaseIteration<0_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::SyncPhaseA,
                             tmpl::list<Actions::RecordPhaseIteration<1_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::SyncPhaseB,
                             tmpl::list<Actions::RecordPhaseIteration<2_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<
              Actions::RecordPhaseIteration<3_st>,
              Actions::TerminateAndRestart<ComponentBeta<Metavariables>, 2_st>,
              Actions::RequestPhaseEntryFunction, Actions::StopForSync>>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<Actions::RecordCurrentPhase>(
        Parallel::get_parallel_component<ComponentAlpha>(local_cache),
        next_phase);
    if (next_phase == Metavariables::Phase::Finalize) {
      Parallel::simple_action<Actions::Finalize>(
          Parallel::get_parallel_component<ComponentAlpha>(local_cache));
    } else {
      Parallel::get_parallel_component<ComponentAlpha>(local_cache)
          .start_phase(next_phase);
    }
  }
};

namespace Tags {
struct PhaseRecord : db::SimpleTag {
  using type = std::string;
};
struct Step : db::SimpleTag {
  using type = size_t;
};
}  // namespace Tags

namespace Actions {

struct InitializePhaseRecord {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        Initialization::merge_into_databox<
            InitializePhaseRecord,
            db::AddSimpleTags<Tags::PhaseRecord, Tags::Step>,
            db::AddComputeTags<>, Initialization::MergePolicy::Overwrite>(
            std::move(box), "", 0_st));
  }
};

// simple action called at the start of a phase from execute_next_phase
struct RecordCurrentPhase {
  template <
      typename ParallelComponent, typename DbTagsList, typename ArrayIndex,
      typename Metavariables,
      Requires<tmpl::list_contains_v<DbTagsList, Tags::PhaseRecord>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const typename Metavariables::Phase& phase) noexcept {
    db::mutate<Tags::PhaseRecord>(
        make_not_null(&box),
        [&phase](const gsl::not_null<std::string*> phase_log) noexcept {
          *phase_log +=
              "Entering phase: " + Metavariables::phase_to_string(phase) + "\n";
        });
  }

  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex, typename Metavariables,
            Requires<not tmpl::list_contains_v<DbTagsList, Tags::PhaseRecord>> =
                nullptr>
  static void apply(const db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const typename Metavariables::Phase& /*phase*/) noexcept {}
};

// iterable action called during phases
template <size_t phase>
struct RecordPhaseIteration {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::PhaseRecord>(
        make_not_null(&box),
        [](const gsl::not_null<std::string*> phase_log) noexcept {
          *phase_log += "Running phase: " +
                        Metavariables::phase_to_string(
                            static_cast<typename Metavariables::Phase>(phase)) +
                        "\n";
        });
    if (static_cast<typename Metavariables::Phase>(phase) ==
        Metavariables::Phase::Evolve) {
      db::mutate<Tags::Step>(
          make_not_null(&box),
          [](const gsl::not_null<size_t*> step) noexcept { ++(*step); });
    }
    return std::make_tuple(std::move(box));
  }
};

template <typename ComponentToRestart>
struct RestartMe {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename Metavariables>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) noexcept {
    Parallel::get_parallel_component<ComponentToRestart>(cache)
        .perform_algorithm(true);
  }
};

template <typename OtherComponent, size_t interval>
struct TerminateAndRestart {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (db::get<Tags::Step>(box) % interval == 0) {
      Parallel::simple_action<Actions::RestartMe<ParallelComponent>>(
          Parallel::get_parallel_component<OtherComponent>(cache));

      db::mutate<Tags::PhaseRecord>(
          make_not_null(&box),
          [](const gsl::not_null<std::string*> phase_log) noexcept {
            *phase_log += "Terminate and Restart\n";
          });
      return std::make_tuple(std::move(box), true);
    }
    return std::make_tuple(std::move(box), false);
  }
};

struct RequestPhaseEntryFunction {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // see explanation of test steps preceding `TestMetavariables` declaration
    // -- this action is only used for component 'beta'
    const size_t step = db::get<Tags::Step>(box);
    if (step == 7_st or step == 15_st or step == 20_st) {
      Parallel::get_parallel_component<ParallelComponent>(cache)
          .request_sync_phase(Metavariables::Phase::SyncPhaseA);
    }
    if (step == 21_st or step == 25_st) {
      Parallel::get_parallel_component<ParallelComponent>(cache)
          .request_sync_phase(Metavariables::Phase::SyncPhaseB);
    }
    return std::make_tuple(std::move(box));
  }
};

struct RequestPhaseActionReturn {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::AlgorithmControl<Metavariables> algorithm_control{};
    const size_t step = db::get<Tags::Step>(box);
    // see explanation of test steps preceding `TestMetavariables` declaration
    // -- this action is only used for component 'beta'
    if (step == 14_st or step == 20_st or step == 25_st) {
      algorithm_control.global_sync_phases =
          std::unordered_set<typename Metavariables::Phase>{};
      (*algorithm_control.global_sync_phases)
          .insert(Metavariables::Phase::SyncPhaseA);
    }
    if (step >= 10_st and step <= 25_st and
        (step % 5_st == 0_st or step % 7_st == 0_st)) {
      if (not static_cast<bool>(algorithm_control.global_sync_phases)) {
        algorithm_control.global_sync_phases =
            std::unordered_set<typename Metavariables::Phase>{};
      }
      (*algorithm_control.global_sync_phases)
          .insert(Metavariables::Phase::SyncPhaseB);
    }
    return std::make_tuple(std::move(box), algorithm_control);
  }
};

struct StopForSync {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Parallel::AlgorithmControl<Metavariables> algorithm_control{};
    if (db::get<Tags::Step>(box) % 7_st == 0_st or
        db::get<Tags::Step>(box) % 5_st == 0_st) {
      algorithm_control.execution_flag =
          Parallel::AlgorithmExecution::SleepForSyncPhases;
    }
    if(db::get<Tags::Step>(box) > 25_st) {
      algorithm_control.execution_flag = Parallel::AlgorithmExecution::Halt;
    }
    return std::make_tuple(std::move(box), algorithm_control);
  }
};

struct Finalize {
  template <
      typename ParallelComponent, typename DbTagsList, typename ArrayIndex,
      typename Metavariables,
      Requires<tmpl::list_contains_v<DbTagsList, Tags::PhaseRecord>> = nullptr>
  static auto apply(const db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) noexcept {
    const std::string& log = db::get<Tags::PhaseRecord>(box);
    SPECTRE_PARALLEL_REQUIRE(
        log == Metavariables::expected_log(tmpl::type_<ParallelComponent>{}));
  }

  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex, typename Metavariables,
            Requires<not tmpl::list_contains_v<DbTagsList, Tags::PhaseRecord>> =
                nullptr>
  static auto apply(const db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) noexcept {
    SPECTRE_PARALLEL_REQUIRE(false);
  }
};
}  // namespace Actions

// two elements: alpha and beta
// terminate and restart alpha on even time steps
// terminate and restart beta on odd time steps divisible by 3
// request sync actions on counts that are divisible by 5 or 7
//
// all events are recorded in a 'log' string that can be checked at the end.
//
// sync action set (A is 'required', B isn't):
//   'alpha'    'beta'
//  5:  none |   none
//  7:   A   |   none
// 10:  none |     B
// 14:  none |  A and B
// 15:   A   |     B
// 20:   A   |  A and B
// 21:   B   |     B
// 25:   B   |  A and B
struct TestMetavariables {
  using component_list = tmpl::list<ComponentAlpha<TestMetavariables>,
                                    ComponentBeta<TestMetavariables>>;

  enum class Phase {
    Initialization,
    SyncPhaseA,
    SyncPhaseB,
    Evolve,
    Finalize,
    Exit
  };

  static bool is_required_sync_phase(const Phase phase) noexcept {
    return phase == Phase::SyncPhaseA;
  }

  static std::string phase_to_string(const Phase phase) noexcept {
    switch (phase) {
      case Phase::Initialization:
        return "Initialization";
      case Phase::SyncPhaseA:
        return "SyncPhaseA";
      case Phase::SyncPhaseB:
        return "SyncPhaseB";
      case Phase::Evolve:
        return "Evolve";
      case Phase::Finalize:
        return "Finalize";
      case Phase::Exit:
        return "Exit";
      default:
        ERROR("phase_to_string: Unknown phase");
    }
  }

  static constexpr Options::String help =
      "An executable for testing basic phase control flow.";

  static std::string repeat(const std::string& input,
                            const size_t times) noexcept {
    std::string output;
    for (size_t i = 0; i < times; ++i) {
      output += input;
    }
    return output;
  }

  static std::string expected_log(
      tmpl::type_<ComponentAlpha<TestMetavariables>> /*meta*/) noexcept {
    return "Running phase: Initialization\n"
           "Entering phase: Evolve\n" +
           repeat(repeat("Running phase: Evolve\n", 2_st) +
                      "Terminate and Restart\n",
                  2_st) +              // step 1-4
           "Running phase: Evolve\n"   // step 5
           "Entering phase: Evolve\n"  // step 5 sync phases
           "Running phase: Evolve\n"   // step 6
           "Terminate and Restart\n"
           "Running phase: Evolve\n"       // step 7
           "Entering phase: SyncPhaseA\n"  // step 7 sync phases
           "Running phase: SyncPhaseA\n"
           "Entering phase: Evolve\n"
           "Running phase: Evolve\n"  // step 8
           "Terminate and Restart\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // step 9-10
           "Terminate and Restart\n"
           "Entering phase: SyncPhaseB\n"  // step 10 sync phases
           "Entering phase: Evolve\n" +
           repeat(repeat("Running phase: Evolve\n", 2_st) +
                      "Terminate and Restart\n",
                  2_st) +  // step 11-14
           repeat(
               "Entering phase: SyncPhaseA\n"  // step 14 and 15 sync phases
               "Running phase: SyncPhaseA\n"
               "Entering phase: SyncPhaseB\n"
               "Entering phase: Evolve\n"
               "Running phase: Evolve\n",  // step 15 and 16
               2_st) +
           "Terminate and Restart\n" +
           repeat(repeat("Running phase: Evolve\n", 2_st) +
                      "Terminate and Restart\n",
                  2_st) +                  // step 17-20
           "Entering phase: SyncPhaseA\n"  // step 20 sync phases
           "Running phase: SyncPhaseA\n"
           "Entering phase: SyncPhaseB\n"
           "Entering phase: Evolve\n"
           "Running phase: Evolve\n"       // step 21
           "Entering phase: SyncPhaseB\n"  // step 21 sync phases
           "Running phase: SyncPhaseB\n"
           "Entering phase: Evolve\n" +
           repeat(
               "Running phase: Evolve\n"
               "Terminate and Restart\n"
               "Running phase: Evolve\n",
               2_st) +                     // step 22-25
           "Entering phase: SyncPhaseA\n"  // step 25 sync phases
           "Running phase: SyncPhaseA\n"
           "Entering phase: SyncPhaseB\n"
           "Running phase: SyncPhaseB\n"
           "Entering phase: Evolve\n"
           "Running phase: Evolve\n"
           "Terminate and Restart\n"
           "Entering phase: Finalize\n";
  }

  static std::string expected_log(
      tmpl::type_<ComponentBeta<TestMetavariables>> /*meta*/) noexcept {
    return "Running phase: Initialization\n"
           "Entering phase: Evolve\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // steps 1-3
           "Terminate and Restart\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // steps 4-5
           "Entering phase: Evolve\n"                 // step 5 sync phases
           "Running phase: Evolve\n"                  // step 6
           "Terminate and Restart\n"
           "Running phase: Evolve\n"       // step 7
           "Entering phase: SyncPhaseA\n"  // step 7 sync phases
           "Running phase: SyncPhaseA\n"
           "Entering phase: Evolve\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // steps 8-9
           "Terminate and Restart\n"
           "Running phase: Evolve\n"       // step 10
           "Entering phase: SyncPhaseB\n"  // step 10 sync phases
           "Running phase: SyncPhaseB\n"
           "Entering phase: Evolve\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // steps 11-12
           "Terminate and Restart\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // steps 13-14
           "Entering phase: SyncPhaseA\n"             // step 14 sync phases
           "Running phase: SyncPhaseA\n"
           "Entering phase: SyncPhaseB\n"
           "Running phase: SyncPhaseB\n"
           "Entering phase: Evolve\n"
           "Running phase: Evolve\n"  // step 15
           "Terminate and Restart\n"
           "Entering phase: SyncPhaseA\n"  // step 15 sync phases
           "Running phase: SyncPhaseA\n"
           "Entering phase: SyncPhaseB\n"
           "Running phase: SyncPhaseB\n"
           "Entering phase: Evolve\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // step 16-18
           "Terminate and Restart\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // step 19-20
           "Entering phase: SyncPhaseA\n"             // step 20 sync phases
           "Running phase: SyncPhaseA\n"
           "Entering phase: SyncPhaseB\n"
           "Running phase: SyncPhaseB\n"
           "Entering phase: Evolve\n" +
           "Running phase: Evolve\n"  // step 21
           "Terminate and Restart\n"
           "Entering phase: SyncPhaseB\n"
           "Running phase: SyncPhaseB\n"
           "Entering phase: Evolve\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // step 22-24
           "Terminate and Restart\n"
           "Running phase: Evolve\n"       // step 25
           "Entering phase: SyncPhaseA\n"  // step 25 sync phases
           "Running phase: SyncPhaseA\n"
           "Entering phase: SyncPhaseB\n"
           "Running phase: SyncPhaseB\n"
           "Entering phase: Evolve\n"
           "Running phase: Evolve\n"  // step 26
           "Entering phase: Finalize\n";
  }

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Finalize;
      case Phase::Finalize:
      case Phase::Exit:
        return Phase::Exit;
      default:
        ERROR("Unknown Phase...");
    }

    return Phase::Exit;
  }
};
}  // namespace

/// [charm_init_funcs_example]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// [charm_init_funcs_example]

/// [charm_main_example]
using charmxx_main_component = Parallel::Main<TestMetavariables>;
/// [charm_main_example]

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
