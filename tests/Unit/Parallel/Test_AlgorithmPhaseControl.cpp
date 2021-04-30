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
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {
struct InitializePhaseRecord;
struct RecordCurrentPhase;
template <size_t phase>
struct RecordPhaseIteration;
template <typename ComponentToRestart>
struct RestartMe;
template <typename OtherComponent, size_t interval>
struct TerminateAndRestart;
struct Finalize;
}  // namespace Actions

namespace Tags {
struct PhaseRecord : db::SimpleTag {
  using type = std::string;
};
struct Step : db::SimpleTag {
  using type = size_t;
};
}  // namespace Tags

template <typename TriggerRegistrars>
struct TempPhaseATrigger;

template <typename TriggerRegistrars>
struct TempPhaseBTrigger;

namespace Registrars {
using TempPhaseATrigger = Registration::Registrar<TempPhaseATrigger>;
using TempPhaseBTrigger = Registration::Registrar<TempPhaseBTrigger>;
}  // namespace Registrars

template <typename Metavariables>
struct ComponentAlpha;

template <typename Metavariables>
struct ComponentBeta;

template <typename TriggerRegistrars =
              tmpl::list<Registrars::TempPhaseATrigger>>
struct TempPhaseATrigger : public Trigger<TriggerRegistrars> {
  TempPhaseATrigger() = default;
  explicit TempPhaseATrigger(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TempPhaseATrigger);  // NOLINT

  static constexpr Options::String help{
    "Trigger for going to TempPhaseA."};
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<Tags::Step>;

  bool operator()(const size_t step) const noexcept {
    return step % 5 == 0;
  }
};

template <typename TriggerRegistrars =
              tmpl::list<Registrars::TempPhaseBTrigger>>
struct TempPhaseBTrigger : public Trigger<TriggerRegistrars> {
  TempPhaseBTrigger() = default;
  explicit TempPhaseBTrigger(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TempPhaseBTrigger);  // NOLINT

  static constexpr Options::String help{"Trigger for going to TempPhaseB."};
  using options = tmpl::list<>;

  using argument_tags = tmpl::list<Tags::Step>;

  bool operator()(const size_t step) const noexcept {
    return step % 3 == 0;
  }
};

template <typename TriggerRegistrars>
PUP::able::PUP_ID TempPhaseBTrigger<TriggerRegistrars>::my_PUP_ID = 0;
template <typename TriggerRegistrars>
PUP::able::PUP_ID TempPhaseATrigger<TriggerRegistrars>::my_PUP_ID = 0;

template <typename Metavariables>
struct ComponentAlpha {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::InitializePhaseRecord,
                                        Actions::RecordPhaseIteration<0_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::TempPhaseA,
                             tmpl::list<Actions::RecordPhaseIteration<1_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::TempPhaseB,
                             tmpl::list<Actions::RecordPhaseIteration<2_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<
              Actions::RecordPhaseIteration<3_st>,
              Actions::TerminateAndRestart<ComponentBeta<Metavariables>, 2_st>,
              PhaseControl::Actions::ExecutePhaseChange<
                  typename Metavariables::phase_changes,
                  typename Metavariables::triggers>>>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
      /*initialization_items*/) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& array_proxy =
        Parallel::get_parallel_component<ComponentAlpha>(local_cache);

    array_proxy[0].insert(global_cache, {}, 0);
    array_proxy.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::Finalize) {
      Parallel::simple_action<Actions::Finalize>(
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
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::InitializePhaseRecord,
                                        Actions::RecordPhaseIteration<0_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::TempPhaseA,
                             tmpl::list<Actions::RecordPhaseIteration<1_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::TempPhaseB,
                             tmpl::list<Actions::RecordPhaseIteration<2_st>,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<
              Actions::RecordPhaseIteration<3_st>,
              Actions::TerminateAndRestart<ComponentAlpha<Metavariables>, 3_st>,
              PhaseControl::Actions::ExecutePhaseChange<
                  typename Metavariables::phase_changes,
                  typename Metavariables::triggers>>>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::Finalize) {
      Parallel::simple_action<Actions::Finalize>(
          Parallel::get_parallel_component<ComponentBeta>(local_cache));
    } else {
      Parallel::get_parallel_component<ComponentBeta>(local_cache)
          .start_phase(next_phase);
    }
  }
};

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

// iterable action called during phases
template <size_t Phase>
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
                        Metavariables::phase_name(
                            static_cast<typename Metavariables::Phase>(Phase)) +
                        "\n";
        });
    if (static_cast<typename Metavariables::Phase>(Phase) ==
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
      if(db::get<Tags::Step>(box) <= 15) {
        Parallel::simple_action<Actions::RestartMe<ParallelComponent>>(
            Parallel::get_parallel_component<OtherComponent>(cache));

        db::mutate<Tags::PhaseRecord>(
            make_not_null(&box),
            [](const gsl::not_null<std::string*> phase_log) noexcept {
              *phase_log += "Terminate and Restart\n";
            });
        return std::make_tuple(std::move(box),
                               Parallel::AlgorithmExecution::Pause);
      } else {
        db::mutate<Tags::PhaseRecord>(
            make_not_null(&box),
            [](const gsl::not_null<std::string*> phase_log) noexcept {
              *phase_log += "Terminate Completion\n";
            });
        return std::make_tuple(std::move(box),
                               Parallel::AlgorithmExecution::Halt);
      }
    }
    return std::make_tuple(std::move(box),
                           Parallel::AlgorithmExecution::Continue);
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
// action request pattern:
//   'alpha'    'beta'
//  5:  none |   none
//  7:   A   |   none
// 10:  none |     B
// 14:  none |  A and B
struct TestMetavariables {
  using component_list = tmpl::list<ComponentAlpha<TestMetavariables>,
                                    ComponentBeta<TestMetavariables>>;

  enum class Phase {
    Initialization,
    TempPhaseA,
    TempPhaseB,
    Evolve,
    Finalize,
    Exit
  };

  using triggers =
      tmpl::list<Registrars::TempPhaseATrigger, Registrars::TempPhaseBTrigger>;
  using phase_changes =
      tmpl::list<PhaseControl::Registrars::VisitAndReturn<TestMetavariables,
                                                          Phase::TempPhaseA>,
                 PhaseControl::Registrars::VisitAndReturn<TestMetavariables,
                                                          Phase::TempPhaseB>>;

  using phase_change_tags_and_combines_list =
      PhaseControl::get_phase_change_tags<phase_changes>;

  using initialize_phase_change_decision_data =
      PhaseControl::InitializePhaseChangeDecisionData<phase_changes, triggers>;

  using const_global_cache_tags = tmpl::list<
      PhaseControl::Tags::PhaseChangeAndTriggers<phase_changes, triggers>>;

  static std::string phase_name(const Phase phase) noexcept {
    switch (phase) {
      case Phase::Initialization:
        return "Initialization";
      case Phase::TempPhaseA:
        return "TempPhaseA";
      case Phase::TempPhaseB:
        return "TempPhaseB";
      case Phase::Evolve:
        return "Evolve";
      case Phase::Finalize:
        return "Finalize";
      case Phase::Exit:
        return "Exit";
      default:
        ERROR("phase_name: Unknown phase");
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
    return "Running phase: Initialization\n" +
           repeat("Running phase: Evolve\n", 2_st) +
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 3 -> B
           "Running phase: TempPhaseB\n"
           "Running phase: Evolve\n"  // step 4
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 5 -> A
           "Running phase: TempPhaseA\n"
           "Running phase: Evolve\n"  // step 6 -> B
           "Terminate and Restart\n"
           "Running phase: TempPhaseB\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // step 7-8
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 9 -> B
           "Running phase: TempPhaseB\n"
           "Running phase: Evolve\n"  // step 10 -> A
           "Terminate and Restart\n"
           "Running phase: TempPhaseA\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // step 11-12 -> B
           "Terminate and Restart\n"
           "Running phase: TempPhaseB\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // step 13-14
           "Terminate and Restart\n"
           "Running phase: Evolve\n"  // step 15 -> B then A
           "Running phase: TempPhaseA\n"
           "Running phase: TempPhaseB\n"
           "Running phase: Evolve\n"  // step 16
           "Terminate Completion\n";
  }

  static std::string expected_log(
      tmpl::type_<ComponentBeta<TestMetavariables>> /*meta*/) noexcept {
    return "Running phase: Initialization\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // steps 1-3 -> B
           "Terminate and Restart\n"
           "Running phase: TempPhaseB\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // steps 4-5 -> A
           "Running phase: TempPhaseA\n"
           "Running phase: Evolve\n"  // step 6 -> B
           "Terminate and Restart\n"
           "Running phase: TempPhaseB\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // steps 8-9 -> B
           "Terminate and Restart\n"
           "Running phase: TempPhaseB\n"
           "Running phase: Evolve\n"  // step 10 -> A
           "Running phase: TempPhaseA\n" +
           repeat("Running phase: Evolve\n", 2_st) +  // steps 11-12 -> B
           "Terminate and Restart\n"
           "Running phase: TempPhaseB\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // steps 13-15 -> A then B
           "Terminate and Restart\n"
           "Running phase: TempPhaseA\n"
           "Running phase: TempPhaseB\n" +
           repeat("Running phase: Evolve\n", 3_st) +  // steps 16-18
           "Terminate Completion\n";
  }

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          TestMetavariables>& cache_proxy) noexcept {
    const auto next_phase =
        PhaseControl::arbitrate_phase_change<phase_changes, triggers>(
            phase_change_decision_data, current_phase,
            *(cache_proxy.ckLocalBranch()));
    if (next_phase.has_value()) {
      return next_phase.value();
    }
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

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

// [charm_init_funcs_example]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &Parallel::register_derived_classes_with_charm<
        Trigger<TestMetavariables::triggers>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<TestMetavariables::phase_changes>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
// [charm_init_funcs_example]

// [charm_main_example]
using charmxx_main_component = Parallel::Main<TestMetavariables>;
// [charm_main_example]

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
