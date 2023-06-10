// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {
struct InitializePhaseRecord;
struct RecordCurrentPhase;
template <Parallel::Phase Phase>
struct RecordPhaseIteration;
template <typename ComponentToRestart>
struct RestartMe;
template <typename OtherComponent, size_t interval>
struct TerminateAndRestart;
struct Testing;
}  // namespace Actions

namespace Tags {
struct PhaseRecord : db::SimpleTag {
  using type = std::string;
};
struct Step : db::SimpleTag {
  using type = size_t;
};
}  // namespace Tags

template <typename Metavariables>
struct ComponentAlpha;

template <typename Metavariables>
struct ComponentBeta;

struct RegisterTrigger : public Trigger {
  RegisterTrigger() = default;
  explicit RegisterTrigger(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RegisterTrigger);  // NOLINT

  static constexpr Options::String help{"Trigger for going to Register."};
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<Tags::Step>;

  bool operator()(const size_t step) const { return step % 5 == 0; }
};

struct SolveTrigger : public Trigger {
  SolveTrigger() = default;
  explicit SolveTrigger(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SolveTrigger);  // NOLINT

  static constexpr Options::String help{"Trigger for going to Solve."};
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<Tags::Step>;

  bool operator()(const size_t step) const { return step % 3 == 0; }
};

PUP::able::PUP_ID SolveTrigger::my_PUP_ID = 0;
PUP::able::PUP_ID RegisterTrigger::my_PUP_ID = 0;

template <typename Metavariables>
struct ComponentAlpha {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<Actions::InitializePhaseRecord,
                                        Actions::RecordPhaseIteration<
                                            Parallel::Phase::Initialization>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Register,
          tmpl::list<Actions::RecordPhaseIteration<Parallel::Phase::Register>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Solve,
          tmpl::list<Actions::RecordPhaseIteration<Parallel::Phase::Solve>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<
              Actions::RecordPhaseIteration<Parallel::Phase::Evolve>,
              Actions::TerminateAndRestart<ComponentBeta<Metavariables>, 2_st>,
              PhaseControl::Actions::ExecutePhaseChange>>>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const std::unordered_set<size_t>& /*procs_to_ignore*/ = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<ComponentAlpha>(local_cache);

    array_proxy[0].insert(global_cache, tuples::TaggedTuple<>{}, 0);
    array_proxy.doneInserting();
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    if (next_phase == Parallel::Phase::Testing) {
      Parallel::simple_action<Actions::Testing>(
          Parallel::get_parallel_component<ComponentAlpha>(local_cache));
    } else {
      Parallel::get_parallel_component<ComponentAlpha>(local_cache)
          .start_phase(next_phase);
    }
  }
};

template <typename Metavariables>
struct ComponentBeta {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<Actions::InitializePhaseRecord,
                                        Actions::RecordPhaseIteration<
                                            Parallel::Phase::Initialization>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Register,
          tmpl::list<Actions::RecordPhaseIteration<Parallel::Phase::Register>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Solve,
          tmpl::list<Actions::RecordPhaseIteration<Parallel::Phase::Solve>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<Actions::RecordPhaseIteration<Parallel::Phase::Evolve>,
                     PhaseControl::Actions::ExecutePhaseChange,
                     Actions::TerminateAndRestart<ComponentAlpha<Metavariables>,
                                                  3_st>>>>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    if (next_phase == Parallel::Phase::Testing) {
      Parallel::simple_action<Actions::Testing>(
          Parallel::get_parallel_component<ComponentBeta>(local_cache));
    } else {
      Parallel::get_parallel_component<ComponentBeta>(local_cache)
          .start_phase(next_phase);
    }
  }
};

namespace Actions {

struct InitializePhaseRecord {
  using simple_tags = tmpl::list<Tags::PhaseRecord, Tags::Step>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// iterable action called during phases
template <Parallel::Phase Phase>
struct RecordPhaseIteration {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Tags::PhaseRecord>(
        [](const gsl::not_null<std::string*> phase_log) {
          *phase_log += MakeString{} << "Running phase: " << Phase << "\n";
        },
        make_not_null(&box));
    if (Phase == Parallel::Phase::Evolve) {
      db::mutate<Tags::Step>(
          [](const gsl::not_null<size_t*> step) { ++(*step); },
          make_not_null(&box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename ComponentToRestart>
struct RestartMe {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename Metavariables>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    Parallel::get_parallel_component<ComponentToRestart>(cache)
        .perform_algorithm(true);
  }
};

template <typename OtherComponent, size_t interval>
struct TerminateAndRestart {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<Tags::Step>(box) % interval == 0) {
      if(db::get<Tags::Step>(box) < 15) {
        Parallel::simple_action<Actions::RestartMe<ParallelComponent>>(
            Parallel::get_parallel_component<OtherComponent>(cache));

        db::mutate<Tags::PhaseRecord>(
            [](const gsl::not_null<std::string*> phase_log) {
              *phase_log += "Terminate and Restart\n";
            },
            make_not_null(&box));
        return {Parallel::AlgorithmExecution::Pause, std::nullopt};
      } else {
        db::mutate<Tags::PhaseRecord>(
            [](const gsl::not_null<std::string*> phase_log) {
              *phase_log += "Terminate Completion\n";
            },
            make_not_null(&box));
        return {Parallel::AlgorithmExecution::Halt, std::nullopt};
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct Testing {
  template <
      typename ParallelComponent, typename DbTagsList, typename ArrayIndex,
      typename Metavariables,
      Requires<tmpl::list_contains_v<DbTagsList, Tags::PhaseRecord>> = nullptr>
  static auto apply(const db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
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
                    const ArrayIndex& /*array_index*/) {
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
// action request pattern:
//     'alpha'    'beta'
//  5:  none    |   none
//  7: Register |   none
// 10:  none    |     Solve
// 14:  none    |  Register and Solve
struct TestMetavariables {
  using component_list = tmpl::list<ComponentAlpha<TestMetavariables>,
                                    ComponentBeta<TestMetavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            PhaseChange,
            tmpl::list<PhaseControl::VisitAndReturn<Parallel::Phase::Register>,
                       PhaseControl::VisitAndReturn<Parallel::Phase::Solve>>>,
        tmpl::pair<Trigger, tmpl::list<RegisterTrigger, SolveTrigger>>>;
  };

  using const_global_cache_tags =
      tmpl::list<PhaseControl::Tags::PhaseChangeAndTriggers>;

  static constexpr Options::String help =
      "An executable for testing basic phase control flow.";

  static std::string repeat(const std::string& input, const size_t times) {
    std::string output;
    for (size_t i = 0; i < times; ++i) {
      output += input;
    }
    return output;
  }

  static std::string expected_log(
      tmpl::type_<ComponentAlpha<TestMetavariables>> /*meta*/) {
    return "Running phase: Initialization\n" +
           repeat("Running phase: Evolve\n", 2_st) +
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 3 -> Solve
           "Running phase: Solve\n"
           "Running phase: Evolve\n"  // step 4
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 5 -> Register
           "Running phase: Register\n"
           "Running phase: Evolve\n"  // step 6 -> Solve
           "Terminate and Restart\n"
           "Running phase: Solve\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // step 7-8
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 9 -> Solve
           "Running phase: Solve\n"
           "Running phase: Evolve\n"  // step 10 -> Register
           "Terminate and Restart\n"
           "Running phase: Register\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // step 11-12 -> Solve
           "Terminate and Restart\n"
           "Running phase: Solve\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // step 13-14
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 15 -> Solve then Register
           "Running phase: Register\n"
           "Running phase: Solve\n"
           "Running phase: Evolve\n"  // step 16
           "Terminate Completion\n";
  }

  static std::string expected_log(
      tmpl::type_<ComponentBeta<TestMetavariables>> /*meta*/) {
    return "Running phase: Initialization\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // steps 1-3 -> Solve
           "Running phase: Solve\n"
           "Terminate and Restart\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // steps 4-5 -> Register
           "Running phase: Register\n"
           "Running phase: Evolve\n"  // step 6 -> Solve
           "Running phase: Solve\n"
           "Terminate and Restart\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // steps 8-9 -> Solve
           "Running phase: Solve\n"
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 10 -> Register
           "Running phase: Register\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // steps 11-12 -> Solve
           "Running phase: Solve\n"
           "Terminate and Restart\n" +
           repeat("Running phase: Evolve\n",
                  3_st) +  // steps 13-15 -> Register then Solve
           "Running phase: Register\n"
           "Running phase: Solve\n"
           "Terminate Completion\n";
  }

  static constexpr std::array<Parallel::Phase, 4> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Evolve,
       Parallel::Phase::Testing, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

// [charm_init_funcs_example]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &register_factory_classes_with_charm<TestMetavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
// [charm_init_funcs_example]

// [charm_main_example]
using charmxx_main_component = Parallel::Main<TestMetavariables>;
// [charm_main_example]

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
