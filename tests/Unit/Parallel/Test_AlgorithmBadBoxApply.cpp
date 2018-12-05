// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

struct error_size_zero {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {}
};

template <class Metavariables>
struct Component {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<tmpl::list<>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<error_size_zero>(
        Parallel::get_parallel_component<Component>(local_cache));
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

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
