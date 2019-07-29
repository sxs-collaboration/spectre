// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Options/Options.hpp"
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "ParallelBackend/ConstGlobalCache.hpp"
#include "ParallelBackend/InitializationFunctions.hpp"
#include "ParallelBackend/Invoke.hpp"
#include "ParallelBackend/Main.hpp"
#include "ParallelBackend/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

struct another_action {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {}
};

struct error_call_single_action_from_action {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    Parallel::simple_action<another_action>(*(
        Parallel::get_parallel_component<ParallelComponent>(cache).ckLocal()));
  }
};

template <class Metavariables>
struct Component {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<error_call_single_action_from_action>(
        *(Parallel::get_parallel_component<Component>(local_cache).ckLocal()));
  }

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) {}
};

struct TestMetavariables {
  using component_list = tmpl::list<Component<TestMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Initialization, Exit };

  static constexpr OptionString help = "Executable for testing";

  static Phase determine_next_phase(const Phase& /*current_phase*/,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        TestMetavariables>& /*cache_proxy*/) {
    return Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "ParallelBackend/CharmMain.tpp"  // IWYU pragma: keep
