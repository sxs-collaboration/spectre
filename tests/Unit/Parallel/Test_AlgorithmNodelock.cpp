// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

static constexpr int number_of_1d_array_elements_per_core = 10;

struct TestMetavariables;

namespace PUP {
class er;
}  // namespace PUP
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

template <class Metavariables>
struct ArrayParallelComponent;

template <class Metavariables>
struct NodegroupParallelComponent;

namespace Tags {
struct vector_of_array_indexs : db::SimpleTag {
  static std::string name() { return "vector_of_array_indexs"; }
  using type = std::vector<int>;
};

struct total_receives_on_node : db::SimpleTag {
  static std::string name() { return "total_receives_on_node"; }
  using type = int;
};
}  // namespace Tags

struct nodegroup_initialize {
  using simple_tags =
      tmpl::list<Tags::vector_of_array_indexs, Tags::total_receives_on_node>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 NodegroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box),
        std::vector<int>(
            static_cast<size_t>(number_of_1d_array_elements_per_core *
                                sys::procs_on_node(sys::my_node()))),
        0);
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct nodegroup_receive {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Tags::vector_of_array_indexs,
                                              db::DataBox<DbTags>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const int& id_of_array) {
    static_assert(std::is_same_v<ParallelComponent,
                                 NodegroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    db::mutate<Tags::vector_of_array_indexs, Tags::total_receives_on_node>(
        [&id_of_array](const gsl::not_null<std::vector<int>*> array_indexs,
                       const gsl::not_null<int*> total_receives_on_node) {
          if (static_cast<int>(array_indexs->size()) !=
              number_of_1d_array_elements_per_core *
                  sys::procs_on_node(sys::my_node())) {
            (*array_indexs)[static_cast<size_t>(id_of_array)]++;
          }
          std::for_each(array_indexs->begin(), array_indexs->end(),
                        [](int& t) { t++; });
          ++*total_receives_on_node;
        },
        make_not_null(&box));
  }
};

struct nodegroup_check_first_result {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Tags::vector_of_array_indexs,
                                              db::DataBox<DbTags>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 NodegroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<Tags::total_receives_on_node>(box) ==
                             number_of_1d_array_elements_per_core *
                                 sys::procs_on_node(sys::my_node()));
    decltype(auto) v = db::get<Tags::vector_of_array_indexs>(box);
    SPECTRE_PARALLEL_REQUIRE(static_cast<int>(v.size()) ==
                             number_of_1d_array_elements_per_core *
                                 sys::procs_on_node(sys::my_node()));
    std::for_each(v.begin(), v.end(), [](const int& value) {
      SPECTRE_PARALLEL_REQUIRE(value == number_of_1d_array_elements_per_core *
                                            sys::procs_on_node(sys::my_node()));
    });
  }
};

struct nodegroup_threaded_receive {
  // [threaded_action_example]
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Tags::vector_of_array_indexs,
                                              db::DataBox<DbTags>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const int& id_of_array) {
    // [threaded_action_example]
    node_lock->lock();
    db::mutate<Tags::vector_of_array_indexs, Tags::total_receives_on_node>(
        [&id_of_array](const gsl::not_null<std::vector<int>*> array_indexs,
                       const gsl::not_null<int*> total_receives_on_node) {
          if (static_cast<int>(array_indexs->size()) !=
              number_of_1d_array_elements_per_core *
                  sys::procs_on_node(sys::my_node())) {
            (*array_indexs)[static_cast<size_t>(id_of_array)]++;
          }
          std::for_each(array_indexs->begin(), array_indexs->end(),
                        [](int& t) { t++; });
          ++*total_receives_on_node;
        },
        make_not_null(&box));
    node_lock->unlock();
  }
};

struct nodegroup_check_threaded_result {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Tags::vector_of_array_indexs,
                                              db::DataBox<DbTags>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    static_assert(std::is_same_v<ParallelComponent,
                                 NodegroupParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<Tags::total_receives_on_node>(box) ==
                             2 * number_of_1d_array_elements_per_core *
                                 sys::procs_on_node(sys::my_node()));
    decltype(auto) v = db::get<Tags::vector_of_array_indexs>(box);
    SPECTRE_PARALLEL_REQUIRE(static_cast<int>(v.size()) ==
                             number_of_1d_array_elements_per_core *
                                 sys::procs_on_node(sys::my_node()));
    std::for_each(v.begin(), v.end(), [](const int& value) {
      SPECTRE_PARALLEL_REQUIRE(value ==
                               2 * number_of_1d_array_elements_per_core *
                                   sys::procs_on_node(sys::my_node()));
    });
  }
};

struct reduce_to_nodegroup {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    static_assert(std::is_same_v<ParallelComponent,
                                 ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    auto& local_nodegroup = *Parallel::local_branch(
        Parallel::get_parallel_component<
            NodegroupParallelComponent<Metavariables>>(cache));
    Parallel::simple_action<nodegroup_receive>(local_nodegroup, array_index);
  }
};

struct reduce_threaded_method {
  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    static_assert(std::is_same_v<ParallelComponent,
                                 ArrayParallelComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    auto& local_nodegroup = *Parallel::local_branch(
        Parallel::get_parallel_component<
            NodegroupParallelComponent<Metavariables>>(cache));
    Parallel::threaded_action<nodegroup_threaded_receive>(local_nodegroup,
                                                          array_index);
  }
};

template <class Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Register, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(local_cache);

    const size_t number_of_procs = static_cast<size_t>(sys::number_of_procs());
    TestHelpers::Parallel::assign_array_elements_round_robin_style(
        array_proxy,
        static_cast<size_t>(number_of_1d_array_elements_per_core) *
            number_of_procs,
        number_of_procs, {}, global_cache, procs_to_ignore);
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(local_cache);
    if (next_phase == Parallel::Phase::Register) {
      Parallel::simple_action<reduce_to_nodegroup>(array_proxy);
    }
    if (next_phase == Parallel::Phase::Execute) {
      Parallel::simple_action<reduce_threaded_method>(array_proxy);
    }
  }
};

template <class Metavariables>
struct NodegroupParallelComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<nodegroup_initialize>>,
      Parallel::PhaseActions<Parallel::Phase::Register, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Solve, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Execute, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& nodegroup_proxy =
        Parallel::get_parallel_component<NodegroupParallelComponent>(
            local_cache);
    if (next_phase == Parallel::Phase::Solve) {
      Parallel::simple_action<nodegroup_check_first_result>(nodegroup_proxy);
    }
    if (next_phase == Parallel::Phase::Testing) {
      Parallel::simple_action<nodegroup_check_threaded_result>(nodegroup_proxy);
    }
  }
};

struct TestMetavariables {
  using component_list =
      tmpl::list<ArrayParallelComponent<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;

  static constexpr const char* const help{"Test nodelocks in Algorithm"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  static constexpr std::array<Parallel::Phase, 6> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::Solve, Parallel::Phase::Execute,
       Parallel::Phase::Testing, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
