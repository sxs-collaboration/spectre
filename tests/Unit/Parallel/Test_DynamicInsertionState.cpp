// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/CharmMain.tpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/Factory.hpp"
#include "Parallel/PhaseControl/InitializePhaseChangeDecisionData.hpp"
#include "Parallel/PhaseControl/PhaseChange.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// This executable tests that newly created elements are at the same step in
// an iterable action list as the already existing elements when the algorithm
// resumes.

namespace {
constexpr size_t number_of_iterations = 5;
constexpr size_t number_of_doubling_actions = 2;
constexpr size_t expected_value =
    two_to_the(number_of_iterations * number_of_doubling_actions + 1);

struct CheckReduction {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const int& value) {
    SPECTRE_PARALLEL_REQUIRE(expected_value == value);
  }
};

struct Value : db::SimpleTag {
  using type = int;
};

struct Iteration : db::SimpleTag {
  using type = size_t;
};

struct Initialize {
  using simple_tags = tmpl::list<Value, Iteration>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box), 2, 0_st);
    return {Parallel::AlgorithmExecution::Halt, std::nullopt};
  }
};

template <size_t StepNumber>
struct DoubleValue {
  using simple_tags = tmpl::list<Value>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Value>([](const gsl::not_null<int*> value) { *value *= 2; },
                      make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct CheckForTermination {
  using simple_tags = tmpl::list<Value>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Iteration>(
        [](const gsl::not_null<size_t*> iteration) { ++(*iteration); },
        make_not_null(&box));

    return {db::get<Iteration>(box) == number_of_iterations
                ? Parallel::AlgorithmExecution::Halt
                : Parallel::AlgorithmExecution::Continue,
            std::nullopt};
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
    Parallel::ReductionData<Parallel::ReductionDatum<int, funcl::Plus<>>>
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
                    const ArrayIndex& /*array_index*/, const int value,
                    const size_t iteration) {
    db::mutate<Value, Iteration>(
        [&value, &iteration](const gsl::not_null<int*> box_value,
                             const gsl::not_null<size_t*> box_iteration) {
          *box_value = value;
          *box_iteration = iteration;
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
    const int value = db::get<Value>(box) / 2;
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& new_id : new_ids) {
      Parallel::simple_action<SetValue>(array_proxy[new_id], value,
                                        db::get<Iteration>(box));
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
                    int accumulated_value, size_t iteration) {
    accumulated_value += db::get<Value>(box);
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    if (additional_children_ids.empty()) {
      Parallel::simple_action<SetValue>(array_proxy[parent_index],
                                        accumulated_value, iteration);
    } else {
      const auto next_child_id = additional_children_ids.front();
      additional_children_ids.pop_front();
      Parallel::simple_action<CollectDataFromChildren>(
          array_proxy[next_child_id], parent_index, additional_children_ids,
          accumulated_value, iteration);
    }
    array_proxy[array_index].ckDestroy();
  }
};

struct CreateChild {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex, typename ElementProxy,
            typename ElementIndex>
  static void apply(
      db::DataBox<DbTagList>& /*box*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ElementProxy element_proxy,
      ElementIndex parent_id, ElementIndex child_id,
      std::vector<ElementIndex> children_ids,
      std::unordered_map<Parallel::Phase, size_t> phase_bookmarks) {
    auto my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    ASSERT(
        alg::count(children_ids, child_id) == 1,
        "Child " << child_id << " does not exist uniquely in " << children_ids);
    const auto child_it = alg::find(children_ids, child_id);
    if (*child_it == children_ids.back()) {
      auto parent_proxy = element_proxy(parent_id);
      element_proxy(child_id).insert(
          cache.thisProxy, Parallel::Phase::AdjustDomain, phase_bookmarks,
          std::make_unique<Parallel::SimpleActionCallback<
              SendDataToChildren, decltype(parent_proxy),
              std::vector<ElementIndex>>>(parent_proxy,
                                          std::move(children_ids)));
    } else {
      const auto next_child_it = std::next(child_it);
      auto next_child = *next_child_it;
      element_proxy(child_id).insert(
          cache.thisProxy, Parallel::Phase::AdjustDomain, phase_bookmarks,
          std::make_unique<Parallel::SimpleActionCallback<
              CreateChild, decltype(my_proxy), ElementProxy, ElementIndex,
              ElementIndex, std::vector<ElementIndex>,
              std::unordered_map<Parallel::Phase, size_t>>>(
              my_proxy, element_proxy, std::move(parent_id),
              std::move(next_child), std::move(children_ids), phase_bookmarks));
    }
  }
};

struct ChangeArray {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    auto& array_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    auto my_proxy = array_proxy[array_index];

    const auto& phase_bookmarks = Parallel::local(my_proxy)->phase_bookmarks();

    auto create_children = [&cache, &array_index, &array_proxy,
                            &phase_bookmarks]() {
      std::vector new_ids{1, 2};
      auto& singleton_proxy =
          Parallel::get_parallel_component<amr::Component<Metavariables>>(
              cache);
      Parallel::simple_action<CreateChild>(singleton_proxy, array_proxy,
                                           array_index, new_ids.front(),
                                           new_ids, phase_bookmarks);
    };

    auto create_parent = [&box, &cache, &array_proxy, &my_proxy,
                          &phase_bookmarks](std::deque<int> ids_to_join) {
      ids_to_join.pop_front();
      const int parent_id = 0;
      array_proxy(parent_id).insert(
          cache.thisProxy, Parallel::Phase::AdjustDomain, phase_bookmarks,
          std::make_unique<Parallel::SimpleActionCallback<
              CollectDataFromChildren, decltype(my_proxy), int, std::deque<int>,
              int, size_t>>(my_proxy, parent_id, std::move(ids_to_join), 0,
                            db::get<Iteration>(box)));
    };

    if (0 == array_index) {
      create_children();
    } else if (1 == array_index) {
      create_parent({1, 2});
    }
  }
};

template <class Metavariables>
struct TestArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<Initialize>>,
      Parallel::PhaseActions<
          Parallel::Phase::Execute,
          tmpl::list<DoubleValue<0>, PhaseControl::Actions::ExecutePhaseChange,
                     DoubleValue<1>, CheckForTermination>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  using array_allocation_tags = tmpl::list<>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
      /*array_allocation_items*/
      = {},
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<TestArray>(local_cache);

    TestHelpers::Parallel::assign_array_elements_round_robin_style(
        array_proxy, 1_st, static_cast<size_t>(sys::number_of_procs()), {},
        global_cache, procs_to_ignore);
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& my_proxy = Parallel::get_parallel_component<TestArray>(local_cache);
    my_proxy.start_phase(next_phase);
    if (next_phase == Parallel::Phase::AdjustDomain) {
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

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
        tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers>>>;
  };

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
              int, std::vector<int>,
              std::unordered_map<Parallel::Phase, size_t>>,
          Parallel::SimpleActionCallback<
              SendDataToChildren,
              CProxyElement_AlgorithmArray<TestArray<TestMetavariables>, int>,
              std::vector<int>>,
          Parallel::SimpleActionCallback<
              CollectDataFromChildren,
              CProxyElement_AlgorithmArray<TestArray<TestMetavariables>, int>,
              int, std::deque<int>, int, size_t>>{});
}
}  // namespace

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<TestMetavariables>();
  Parallel::charmxx::register_init_node_and_proc(
      {&register_factory_classes_with_charm<TestMetavariables>,
       &register_callback},
      {});
}
