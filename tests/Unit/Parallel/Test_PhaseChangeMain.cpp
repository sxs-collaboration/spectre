// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ContributeToPhaseChangeReduction.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControlReductionHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PUP {
class er;
}  // namespace PUP

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
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 0_st);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct IncrementStep {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<StepNumber>(
        [](const gsl::not_null<size_t*> step_number) { ++(*step_number); },
        make_not_null(&box));
    SPECTRE_PARALLEL_REQUIRE(db::get<StepNumber>(box) < 31);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct ReportArrayPhaseControlDataAndTerminate{
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache, const int array_index,
      ActionList /*meta*/, const ParallelComponent* const /*meta*/) {
    Parallel::contribute_to_phase_change_reduction<
        ArrayComponent<Metavariables>>(
        tuples::TaggedTuple<IsDone>{array_index == 0
                                        ? db::get<StepNumber>(box) % 2 == 0
                                        : db::get<StepNumber>(box) % 3 == 0},
        cache, array_index);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct ReportGroupPhaseControlDataAndTerminate {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Parallel::contribute_to_phase_change_reduction<
        GroupComponent<Metavariables>>(
        tuples::TaggedTuple<IsDone, PhaseChangeStepNumber>{
            false, db::get<StepNumber>(box)},
        cache);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

template <class Metavariables>
struct GroupComponent {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<InitializeStepTag>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<IncrementStep, ReportGroupPhaseControlDataAndTerminate>>>;
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
struct ArrayComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<InitializeStepTag>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<IncrementStep, ReportArrayPhaseControlDataAndTerminate>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayComponent>(local_cache);

    TestHelpers::Parallel::assign_array_elements_round_robin_style(
        array_proxy, 2, static_cast<size_t>(sys::number_of_procs()), {},
        global_cache, procs_to_ignore);
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
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

  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Evolve,
       Parallel::Phase::Exit}};

  struct DummyPhaseChange : public PhaseChange {
    using phase_change_tags_and_combines =
        tmpl::list<PhaseChangeTest::IsDone,
                   PhaseChangeTest::PhaseChangeStepNumber>;
  };

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<PhaseChange, tmpl::list<DummyPhaseChange>>>;
  };

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
