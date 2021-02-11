// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PhaseChangeTest {
template <class Metavariables>
struct ArrayComponent;

template <class Metavariables>
struct GroupComponent;

struct StepNumber : db::SimpleTag {
  using type = size_t;
};

using PhaseChangeStepNumber =
    PhaseControl::TagAndCombine<PhaseChangeTest::StepNumber, funcl::Max<>>;

struct IsDone {
  using type = bool;
  using combine_method = funcl::And<>;
  using main_combine_method = funcl::Or<>;
};

struct InitializeStepTag {
  using simple_tags = tmpl::list<StepNumber>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool> apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 0_st);
    return {std::move(box), true};
  }
};

struct IncrementStep {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<tmpl::list<DbTags...>>&&> apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<StepNumber>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> step_number) noexcept {
          ++(*step_number);
        });
    SPECTRE_PARALLEL_REQUIRE(db::get<StepNumber>(box) < 31);
    return {std::move(box)};
  }
};

struct ReportArrayPhaseControlDataAndTerminate{
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool> apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) noexcept {
    Parallel::contribute_to_phase_change_reduction<
        ArrayComponent<Metavariables>>(
        tuples::TaggedTuple<IsDone>{array_index == 0
                                        ? db::get<StepNumber>(box) % 2 == 0
                                        : db::get<StepNumber>(box) % 3 == 0},
        cache, array_index);
    return {std::move(box), true};
  }
};

struct ReportGroupPhaseControlDataAndTerminate {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool> apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::contribute_to_phase_change_reduction<
        GroupComponent<Metavariables>>(
        tuples::TaggedTuple<IsDone, PhaseChangeStepNumber>{
            false, db::get<StepNumber>(box)},
        cache);
    return {std::move(box), true};
  }
};

template <class Metavariables>
struct GroupComponent {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox, InitializeStepTag>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolution,
          tmpl::list<IncrementStep, ReportGroupPhaseControlDataAndTerminate>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<GroupComponent>(local_cache)
        .start_phase(next_phase);
  }
};

template <class Metavariables>
struct ArrayComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox, InitializeStepTag>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolution,
          tmpl::list<IncrementStep, ReportArrayPhaseControlDataAndTerminate>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
      /*initialization_items*/) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayComponent>(local_cache);

    for (int i = 0, which_proc = 0, number_of_procs = sys::number_of_procs();
         i < 2; ++i) {
      array_proxy[i].insert(global_cache, {}, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
    array_proxy.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ArrayComponent>(local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace PhaseChangeTest

struct TestMetavariables {
  // Two components, array component has two elements, group component has one
  // element (run only on one core).
  // Array component [0] asks to be done on steps divisible by 2
  // Array component [1] asks to be done on steps divisible by 3
  // The group component just submits its step number.
  // Phase moves to Exit phase when the step is greater than 25 and all
  // components with done states say they are done.
  // Test errors if any component reaches step 31.

  using component_list =
      tmpl::list<PhaseChangeTest::ArrayComponent<TestMetavariables>,
                 PhaseChangeTest::GroupComponent<TestMetavariables>>;

  static constexpr Options::String help = "";

  enum class Phase {
    Initialization,
    Evolution,
    Exit
  };

  using phase_change_tags_and_combines_list =
      tmpl::list<PhaseChangeTest::IsDone,
                 PhaseChangeTest::PhaseChangeStepNumber>;
  struct initialize_phase_change_decision_data {
    static void apply(const gsl::not_null<tuples::tagged_tuple_from_typelist<
                          phase_change_tags_and_combines_list>*>
                          phase_change_decision_data,
                      const Parallel::CProxy_GlobalCache<
                          TestMetavariables>& /*cache_proxy*/) noexcept {
      tuples::get<PhaseChangeTest::IsDone>(*phase_change_decision_data) = false;
      tuples::get<PhaseChangeTest::PhaseChangeStepNumber>(
          *phase_change_decision_data) = 0;
    }
  };

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
          phase_change_decision_data,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::Evolution;
      case Phase::Evolution:
        // confirm that the reduction is being performed consistently each step
        SPECTRE_PARALLEL_REQUIRE(
            tuples::get<PhaseChangeTest::IsDone>(*phase_change_decision_data) ==
            (tuples::get<PhaseChangeTest::PhaseChangeStepNumber>(
                 *phase_change_decision_data) %
                 6 ==
             0));
        if (tuples::get<PhaseChangeTest::IsDone>(
                *phase_change_decision_data) and
            tuples::get<PhaseChangeTest::PhaseChangeStepNumber>(
                *phase_change_decision_data) > 25) {
          return Phase::Exit;
        } else {
          tuples::get<PhaseChangeTest::IsDone>(*phase_change_decision_data) =
              false;
          return Phase::Evolution;
        }
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
