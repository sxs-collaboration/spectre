// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <deque>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// This executable tests dynamic insertion and deletion of array chare elements.
//
// In Parallel::Phase::Initialization 92 elements are created with the simple
// item Value.  The action InitializeValue is called which sets the Value to be
// the array index of the element plus one.
//
// In Parallel::Phase::Execute, the simple action ChangeArray is called.
//    - Elements with array index 0 and 1 and prime numbers > 46 are unchanged
//    - Each element with an array index that is a power of 2 is split into a
//      number of children equal to the array index.  Each child will eventually
//      set Value to be (array index / number of children) while the parent will
//      eventually be deleted.
//    - Each element with an array index that is a prime number in the range
//      [3,46] will be joined with those elements whose array index has prime
//      factors less than or equal to it (e.g. 3 will join any element whose
//      array index is 2^m 3^n for some (m,n)).  The newly created parent
//      element will eventually set its Value to the sum of the Values of the
//      children, while the children will eventually be deleted.
//
//   An element that is split will create its children sequentially, with the
//   last created child element executing the callback of the simple action
//   SendDataToChildren.  SendDataToChildren sends each child its share of Value
//   by calling the simple action SetValue and then deletes the parent element.
//
//   The prime member of joining elements will create the new parent element and
//   execute the callback CollectDataFromChildren on the prime member.
//   CollectDataFromChildren will collect data from the child, either call
//   CollectDataFromChildren on the next child or (when executing on the last
//   child) call SetValue on the new parent element, and then delete the child
//   element.
//
// In Parallel::Phase::Testing, the Values are summed over all the array
// elements via the reduction action ArrayReduce which executes the
// CheckReduction action on the singleton component.  The test succeeds if the
// sum over values of the array has been preserved.

namespace {
static constexpr int initial_number_of_1d_array_elements = 92;
static const double expected_value = 0.5 * initial_number_of_1d_array_elements *
                                     (initial_number_of_1d_array_elements + 1);

struct CheckReduction {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const double& value) {
    SPECTRE_PARALLEL_REQUIRE(expected_value == value);
  }
};

struct Value : db::SimpleTag {
  using type = double;
};

struct InitializeValue {
  using simple_tags = tmpl::list<Value>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               array_index + 1.0);
    return {Parallel::AlgorithmExecution::Halt, std::nullopt};
  }
};

struct ArrayReduce {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    const auto& my_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index];
    const auto& singleton_proxy =
        Parallel::get_parallel_component<amr::Component<Metavariables>>(cache);
    Parallel::ReductionData<Parallel::ReductionDatum<double, funcl::Plus<>>>
        reduction_data{db::get<Value>(box)};
    Parallel::contribute_to_reduction<CheckReduction>(reduction_data, my_proxy,
                                                      singleton_proxy);
  }
};

struct SetValue {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const double value) {
    db::mutate<Value>(
        [&value](const gsl::not_null<double*> box_value) {
          *box_value = value;
        },
        make_not_null(&box));
  }
};

struct SendDataToChildren {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const std::vector<ArrayIndex>& new_ids) {
    const double value = db::get<Value>(box) / new_ids.size();
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& new_id : new_ids) {
      Parallel::simple_action<SetValue>(array_proxy[new_id], value);
    }
    array_proxy[array_index].ckDestroy();
  }
};

struct CollectDataFromChildren {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const ArrayIndex& parent_index,
                    std::deque<ArrayIndex> additional_children_ids,
                    double accumulated_value) {
    accumulated_value += db::get<Value>(box);
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    if (additional_children_ids.empty()) {
      Parallel::simple_action<SetValue>(array_proxy[parent_index],
                                        accumulated_value);
    } else {
      const auto next_child_id = additional_children_ids.front();
      additional_children_ids.pop_front();
      Parallel::simple_action<CollectDataFromChildren>(
          array_proxy[next_child_id], parent_index, additional_children_ids,
          accumulated_value);
    }
    array_proxy[array_index].ckDestroy();
  }
};

struct CreateChild {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex, typename ElementProxy,
            typename ElementIndex>
  static void apply(db::DataBox<DbTagList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    ElementProxy element_proxy, ElementIndex parent_id,
                    ElementIndex child_id,
                    std::vector<ElementIndex> children_ids) {
    auto my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    ASSERT(
        alg::count(children_ids, child_id) == 1,
        "Child " << child_id << " does not exist uniquely in " << children_ids);
    const auto child_it = alg::find(children_ids, child_id);
    if (*child_it == children_ids.back()) {
      auto parent_proxy = element_proxy(parent_id);
      element_proxy(child_id).insert(
          cache.thisProxy, Parallel::Phase::AdjustDomain,
          std::make_unique<Parallel::SimpleActionCallback<
              SendDataToChildren, decltype(parent_proxy),
              std::vector<ElementIndex>>>(parent_proxy,
                                          std::move(children_ids)));
    } else {
      const auto next_child_it = std::next(child_it);
      auto next_child = *next_child_it;
      element_proxy(child_id).insert(
          cache.thisProxy, Parallel::Phase::AdjustDomain,
          std::make_unique<Parallel::SimpleActionCallback<
              CreateChild, decltype(my_proxy), ElementProxy, ElementIndex,
              ElementIndex, std::vector<ElementIndex>>>(
              my_proxy, std::move(element_proxy), std::move(parent_id),
              std::move(next_child), std::move(children_ids)));
    }
  }
};

struct ChangeArray {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    auto my_proxy = array_proxy[array_index];
    auto create_children = [&cache, &array_index,
                            &array_proxy](const size_t number_of_new_elements) {
      std::vector<int> new_ids(number_of_new_elements);
      std::iota(new_ids.begin(), new_ids.end(), 100 * array_index);
      auto& singleton_proxy =
          Parallel::get_parallel_component<amr::Component<Metavariables>>(
              cache);
      Parallel::simple_action<CreateChild>(
          singleton_proxy, array_proxy, array_index, new_ids.front(), new_ids);
    };

    auto create_parent = [&cache, &array_proxy,
                          &my_proxy](std::deque<int> ids_to_join) {
      int id_of_first_child = ids_to_join.front();
      ids_to_join.pop_front();
      int parent_id = 100 * id_of_first_child;
      array_proxy(parent_id).insert(
          cache.thisProxy, Parallel::Phase::Execute,
          std::make_unique<Parallel::SimpleActionCallback<
              CollectDataFromChildren, decltype(my_proxy), int, std::deque<int>,
              double>>(my_proxy, std::move(parent_id), std::move(ids_to_join),
                       0.0));
    };

    std::vector<size_t> ids_to_split{2, 4, 8, 16, 32, 64};
    std::vector<size_t> ids_to_join{3,  5,  7,  11, 13, 17, 19,
                                    23, 29, 31, 37, 41, 43};

    for (const auto id_to_split : ids_to_split) {
      if (id_to_split == static_cast<size_t>(array_index)) {
        create_children(id_to_split);
      }
    }

    if (3 == array_index) {
      create_parent({3, 6, 9, 12, 18, 24, 27, 36, 48, 54, 72, 81});
    } else if (5 == array_index) {
      create_parent({5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 75, 80, 90});
    } else if (7 == array_index) {
      create_parent({7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 84});
    } else if (11 == array_index) {
      create_parent({11, 22, 33, 44, 55, 66, 77, 88});
    } else if (13 == array_index) {
      create_parent({13, 26, 39, 52, 65, 78, 91});
    } else if (17 == array_index) {
      create_parent({17, 34, 51, 68, 85});
    } else if (19 == array_index) {
      create_parent({19, 38, 57, 76});
    } else if (23 == array_index) {
      create_parent({23, 46, 69});
    } else if (29 == array_index) {
      create_parent({29, 58, 87});
    } else if (31 == array_index) {
      create_parent({31, 62});
    } else if (37 == array_index) {
      create_parent({37, 74});
    } else if (41 == array_index) {
      create_parent({41, 82});
    } else if (43 == array_index) {
      create_parent({43, 86});
    }
  }
};

template <class Metavariables>
struct TestArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        tmpl::list<InitializeValue>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<TestArray>(local_cache);

    TestHelpers::Parallel::assign_array_elements_round_robin_style(
        array_proxy, static_cast<size_t>(initial_number_of_1d_array_elements),
        static_cast<size_t>(sys::number_of_procs()), {}, global_cache,
        procs_to_ignore);
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& my_proxy = Parallel::get_parallel_component<TestArray>(local_cache);
    my_proxy.start_phase(next_phase);
    if (next_phase == Parallel::Phase::Execute) {
      Parallel::simple_action<ChangeArray>(my_proxy);
    }
    if (next_phase == Parallel::Phase::Testing) {
      Parallel::simple_action<ArrayReduce>(my_proxy);
    }
  }
};

struct TestMetavariables {
  using component_list = tmpl::list<amr::Component<TestMetavariables>,
                                    TestArray<TestMetavariables>>;

  static constexpr Options::String help =
      "An executable for testing the dynamic insertion and deletion of "
      "elements of an array parallel component";

  static constexpr std::array<Parallel::Phase, 4> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Execute,
       Parallel::Phase::Testing, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

void register_callback() {
  register_classes_with_charm(
      tmpl::list<
          Parallel::SimpleActionCallback<
              CreateChild,
              CProxy_AlgorithmSingleton<amr::Component<TestMetavariables>, int>,
              CProxy_AlgorithmArray<TestArray<TestMetavariables>, int>, int,
              int, std::vector<int>>,
          Parallel::SimpleActionCallback<
              SendDataToChildren,
              CProxyElement_AlgorithmArray<TestArray<TestMetavariables>, int>,
              std::vector<int>>,
          Parallel::SimpleActionCallback<
              CollectDataFromChildren,
              CProxyElement_AlgorithmArray<TestArray<TestMetavariables>, int>,
              int, std::deque<int>, double>>{});
}
}  //  namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &register_callback};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"
