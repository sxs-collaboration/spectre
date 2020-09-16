// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

struct another_action {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {}
};

struct error_call_single_action_from_action {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    Parallel::simple_action<another_action>(*(
        Parallel::get_parallel_component<ParallelComponent>(cache).ckLocal()));
  }
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
      auto& local_cache = *(global_cache.ckLocalBranch());
      Parallel::simple_action<error_call_single_action_from_action>(*(
          Parallel::get_parallel_component<Component>(local_cache).ckLocal()));
    }
  }
};

struct TestMetavariables {
  using component_list = tmpl::list<Component<TestMetavariables>>;

  enum class Phase { Initialization, Execute, Exit };

  static constexpr Options::String help = "Executable for testing";

  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_GlobalCache<
                                        TestMetavariables>& /*cache_proxy*/) {
    if (current_phase == Phase::Initialization) {
      return Phase::Execute;
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
