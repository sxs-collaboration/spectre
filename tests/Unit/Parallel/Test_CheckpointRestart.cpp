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
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
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
  static std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool> apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const std::string component_name =
        pretty_type::short_name<ParallelComponent>() + " " +
        std::to_string(static_cast<int>(array_index));
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box),
        component_name + " invoked action InitializeLog\n");
    return {std::move(box), true};
  }
};

struct MutateLog {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool> apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::Log>(
        make_not_null(&box),
        [&array_index](const gsl::not_null<std::string*> log) noexcept {
          const std::string component_name =
              pretty_type::short_name<ParallelComponent>() + " " +
              std::to_string(static_cast<int>(array_index));
          log->append(component_name + " invoked action MutateLog\n");
        });
    return {std::move(box), true};
  }
};

struct CheckLog {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool> apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const std::string& log = db::get<Tags::Log>(box);
    const std::string component_name =
        pretty_type::short_name<ParallelComponent>() + " " +
        std::to_string(static_cast<int>(array_index));
    const std::string expected_log =
        component_name + " invoked action InitializeLog\n" + component_name +
        " invoked action MutateLog\n";
    SPECTRE_PARALLEL_REQUIRE(log == expected_log);
    return {std::move(box), true};
  }
};

template <class Metavariables>
struct ArrayComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::SetupDataBox, InitializeLog>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::MutateDatabox,
                             tmpl::list<MutateLog>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::CheckDatabox,
                             tmpl::list<CheckLog>>>;
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

template <class Metavariables>
struct GroupComponent {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::SetupDataBox, InitializeLog>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::MutateDatabox,
                             tmpl::list<MutateLog>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::CheckDatabox,
                             tmpl::list<CheckLog>>>;
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
struct NodegroupComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::SetupDataBox, InitializeLog>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::MutateDatabox,
                             tmpl::list<MutateLog>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::CheckDatabox,
                             tmpl::list<CheckLog>>>;
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
struct SingletonComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<Actions::SetupDataBox, InitializeLog>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::MutateDatabox,
                             tmpl::list<MutateLog>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::CheckDatabox,
                             tmpl::list<CheckLog>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
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

  enum class Phase {
    Initialization,
    MutateDatabox,
    WriteCheckpoint,
    CheckDatabox,
    Exit
  };

  using phase_change_tags_and_combines_list = tmpl::list<>;
  struct initialize_phase_change_decision_data {
    static void apply(
        const gsl::not_null<
            tuples::TaggedTuple<>*> /*phase_change_decision_data*/,
        const Parallel::GlobalCache<TestMetavariables>& /*cache*/) noexcept {}
  };

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<tuples::TaggedTuple<Tags...>*>
      /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::MutateDatabox;
      case Phase::MutateDatabox:
        return Phase::WriteCheckpoint;
      case Phase::WriteCheckpoint:
        return Phase::CheckDatabox;
      case Phase::CheckDatabox:
        return Phase::Exit;
      default:
        ERROR("Unknown Phase...");
    }
    return Phase::Exit;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
