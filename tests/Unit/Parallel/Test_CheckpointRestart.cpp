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
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
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

namespace CheckpointTest {
namespace Tags {
struct Log : db::SimpleTag {
  using type = std::string;
};
}  // namespace Tags

struct InitializeLog {
  using simple_tags = tmpl::list<Tags::Log>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const std::string component_name =
        pretty_type::name<ParallelComponent>() + " " +
        std::to_string(static_cast<int>(array_index));
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box),
        component_name + " invoked action InitializeLog\n");
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct MutateLog {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Tags::Log>(
        [&array_index](const gsl::not_null<std::string*> log) {
          const std::string component_name =
              pretty_type::short_name<ParallelComponent>() + " " +
              std::to_string(static_cast<int>(array_index));
          log->append(component_name + " invoked action MutateLog\n");
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct CheckLog {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const std::string& log = db::get<Tags::Log>(box);
    const std::string component_name =
        pretty_type::short_name<ParallelComponent>() + " " +
        std::to_string(static_cast<int>(array_index));
    const std::string expected_log =
        component_name + " invoked action InitializeLog\n" + component_name +
        " invoked action MutateLog\n";
    SPECTRE_PARALLEL_REQUIRE(log == expected_log);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

template <class Metavariables>
struct ArrayComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<InitializeLog>>,
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<MutateLog>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<CheckLog>>>;
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

template <class Metavariables>
struct GroupComponent {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<InitializeLog>>,
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<MutateLog>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<CheckLog>>>;
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
struct NodegroupComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<InitializeLog>>,
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<MutateLog>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<CheckLog>>>;
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
struct SingletonComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<InitializeLog>>,
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<MutateLog>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<CheckLog>>>;
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
}  // namespace CheckpointTest

struct TestMetavariables {
  using component_list =
      tmpl::list<CheckpointTest::ArrayComponent<TestMetavariables>,
                 CheckpointTest::GroupComponent<TestMetavariables>,
                 CheckpointTest::NodegroupComponent<TestMetavariables>,
                 CheckpointTest::SingletonComponent<TestMetavariables>>;

  static constexpr Options::String help = "";

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Execute,
       Parallel::Phase::WriteCheckpoint, Parallel::Phase::Testing,
       Parallel::Phase::Exit}};


  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
