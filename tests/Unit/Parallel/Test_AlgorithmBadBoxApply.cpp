// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PUP {
class er;
}  // namespace PUP
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

struct error_size_zero {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) == 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {}
};

template <class Metavariables>
struct Component {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>&
          global_cache) noexcept {
    if (next_phase == Metavariables::Phase::Execute) {
      auto& local_cache = *Parallel::local_branch(global_cache);
      Parallel::simple_action<error_size_zero>(
          Parallel::get_parallel_component<Component>(local_cache));
    }
  }
};

struct TestMetavariables {
  using component_list = tmpl::list<Component<TestMetavariables>>;

  enum class Phase { Initialization, Execute, Exit };

  static constexpr Options::String help = "Executable for testing";

  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<
          tuples::TaggedTuple<Tags...>*> /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    if (current_phase == Phase::Initialization) {
      return Phase::Execute;
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
