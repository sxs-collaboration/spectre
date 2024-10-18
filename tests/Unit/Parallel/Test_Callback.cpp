// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/CharmMain.tpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Value : db::SimpleTag {
  using type = double;
};

struct TimesIterableActionCalled : db::SimpleTag {
  using type = int;
};

struct InitializeValue {
  using simple_tags = tmpl::list<Value, TimesIterableActionCalled>;
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
                                               array_index + 1.0, 0);
    return {Parallel::AlgorithmExecution::Halt, std::nullopt};
  }
};

struct IncrementValue {
  // Simple action call
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    db::mutate<Value>([](const gsl::not_null<double*> value) { *value += 1.0; },
                      make_not_null(&box));
  }

  // Threaded action call
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/) {
    db::mutate<Value>([](const gsl::not_null<double*> value) { *value += 1.0; },
                      make_not_null(&box));
  }
};

struct MultiplyValueByFactor {
  // Simple action call
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const double factor) {
    db::mutate<Value>(
        [&factor](const gsl::not_null<double*> value) { *value *= factor; },
        make_not_null(&box));
  }

  // Threaded action call
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const double factor) {
    db::mutate<Value>(
        [&factor](const gsl::not_null<double*> value) { *value *= factor; },
        make_not_null(&box));
  }
};

struct DoubleValueOfElement0 {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<TimesIterableActionCalled>(
        [](const gsl::not_null<int*> counter) { ++(*counter); },
        make_not_null(&box));
    if (array_index == 0) {
      db::mutate<Value>(
          [](const gsl::not_null<double*> value) { *value *= 2.0; },
          make_not_null(&box));
      if (db::get<TimesIterableActionCalled>(box) < 5) {
        if constexpr (Parallel::is_array_v<ParallelComponent>) {
          return {Parallel::AlgorithmExecution::Retry, std::nullopt};
        } else if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
          return {Parallel::AlgorithmExecution::Halt, std::nullopt};
        } else {
          ERROR(
              "Only know how to handle arrays and nodegroups in "
              "DoubleValueOfElement0");
        }
      }
    }
    return {Parallel::AlgorithmExecution::Halt, std::nullopt};
  }
};

struct CheckValue {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& array_index) {
    const double value = db::get<Value>(box);
    const int counter = db::get<TimesIterableActionCalled>(box);
    CAPTURE_FOR_ERROR(value);
    CAPTURE_FOR_ERROR(counter);
    if (array_index == 0) {
      if constexpr (Parallel::is_array_v<ParallelComponent>) {
        SPECTRE_PARALLEL_REQUIRE(value == 32.0);
        SPECTRE_PARALLEL_REQUIRE(counter == 5);
      } else if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
        SPECTRE_PARALLEL_REQUIRE(value == 14.0);
        SPECTRE_PARALLEL_REQUIRE(counter == 1);
      } else {
        ERROR("Only know how to handle arrays and nodegroups in CheckValue");
      }
    } else {
      SPECTRE_PARALLEL_REQUIRE(counter == 1);
      SPECTRE_PARALLEL_REQUIRE(value == (array_index == 1 ? 6.0 : 27.0));
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
                                        tmpl::list<InitializeValue>>,
                 Parallel::PhaseActions<Parallel::Phase::Execute,
                                        tmpl::list<DoubleValueOfElement0>>>;
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
    size_t number_of_elements = 3;
    TestHelpers::Parallel::assign_array_elements_round_robin_style(
        array_proxy, number_of_elements,
        static_cast<size_t>(sys::number_of_procs()), {}, global_cache,
        procs_to_ignore);
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& my_proxy = Parallel::get_parallel_component<TestArray>(local_cache);
    my_proxy.start_phase(next_phase);
    if (next_phase == Parallel::Phase::Testing) {
      Parallel::simple_action<CheckValue>(my_proxy);
    }
  }
};

template <class Metavariables>
struct TestNodegroup {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        tmpl::list<InitializeValue>>,
                 Parallel::PhaseActions<Parallel::Phase::Execute,
                                        tmpl::list<DoubleValueOfElement0>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& my_proxy =
        Parallel::get_parallel_component<TestNodegroup>(local_cache);
    my_proxy.start_phase(next_phase);
    if (next_phase == Parallel::Phase::Testing) {
      Parallel::simple_action<CheckValue>(my_proxy);
    }
  }
};

struct RunCallbacks {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    auto& array_proxy =
        Parallel::get_parallel_component<TestArray<Metavariables>>(cache);
    auto proxy_0 = array_proxy[0];
    auto proxy_1 = array_proxy[1];
    auto proxy_2 = array_proxy[2];
    Parallel::PerformAlgorithmCallback<decltype(proxy_0)> callback_0(proxy_0);
    Parallel::SimpleActionCallback<IncrementValue, decltype(proxy_1)>
        callback_1(proxy_1);
    Parallel::SimpleActionCallback<MultiplyValueByFactor, decltype(proxy_2),
                                   double>
        callback_2(proxy_2, 1.5);
    SPECTRE_PARALLEL_REQUIRE(
        callback_0.name().find("PerformAlgorithmCallback") !=
        std::string::npos);
    SPECTRE_PARALLEL_REQUIRE(
        (callback_1.name().find("SimpleActionCallback") != std::string::npos and
         callback_1.name().find("IncrementValue") != std::string::npos));
    SPECTRE_PARALLEL_REQUIRE(
        (callback_2.name().find("SimpleActionCallback") != std::string::npos and
         callback_2.name().find("MultiplyValueByFactor") != std::string::npos));
    callback_0.invoke();
    callback_1.invoke();
    callback_2.invoke();
    auto callback_3 = serialize_and_deserialize(callback_0);
    auto callback_4 = serialize_and_deserialize(callback_1);
    auto callback_5 = serialize_and_deserialize(callback_2);
    SPECTRE_PARALLEL_REQUIRE(callback_0.is_equal_to(callback_3));
    SPECTRE_PARALLEL_REQUIRE_FALSE(callback_0.is_equal_to(callback_4));
    SPECTRE_PARALLEL_REQUIRE(callback_1.is_equal_to(callback_4));
    SPECTRE_PARALLEL_REQUIRE_FALSE(callback_1.is_equal_to(callback_5));
    callback_3.invoke();
    callback_4.invoke();
    callback_5.invoke();
    std::vector<std::unique_ptr<Parallel::Callback>> callbacks;
    callbacks.emplace_back(
        std::make_unique<Parallel::PerformAlgorithmCallback<decltype(proxy_0)>>(
            proxy_0));
    callbacks.emplace_back(
        std::make_unique<
            Parallel::SimpleActionCallback<IncrementValue, decltype(proxy_1)>>(
            proxy_1));
    callbacks.emplace_back(
        std::make_unique<Parallel::SimpleActionCallback<
            MultiplyValueByFactor, decltype(proxy_2), double>>(proxy_2, 2.0));
    SPECTRE_PARALLEL_REQUIRE_FALSE(callback_2.is_equal_to(*callbacks.back()));

    auto& nodegroup_proxy =
        Parallel::get_parallel_component<TestNodegroup<Metavariables>>(cache);
    callbacks.emplace_back(
        std::make_unique<Parallel::ThreadedActionCallback<
            IncrementValue, std::decay_t<decltype(nodegroup_proxy)>>>(
            nodegroup_proxy));
    callbacks.emplace_back(
        std::make_unique<Parallel::ThreadedActionCallback<
            MultiplyValueByFactor, std::decay_t<decltype(nodegroup_proxy)>,
            double>>(nodegroup_proxy, 2.0));
    for (const auto& callback : callbacks) {
      callback->register_with_charm();
      callback->invoke();
    }
    auto pupped_callbacks = serialize_and_deserialize(callbacks);
    for (const auto& callback : pupped_callbacks) {
      callback->invoke();
    }
  }
};

template <class Metavariables>
struct TestSingleton {
  using chare_type = Parallel::Algorithms::Singleton;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& my_proxy =
        Parallel::get_parallel_component<TestSingleton>(local_cache);
    my_proxy.start_phase(next_phase);
    if (next_phase == Parallel::Phase::Execute) {
      Parallel::simple_action<RunCallbacks>(my_proxy);
    }
  }
};

struct TestMetavariables {
  using component_list = tmpl::list<TestSingleton<TestMetavariables>,
                                    TestNodegroup<TestMetavariables>,
                                    TestArray<TestMetavariables>>;

  static constexpr Options::String help =
      "An executable for testing Paralell::Callbacks";

  static constexpr std::array<Parallel::Phase, 4> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Execute,
       Parallel::Phase::Testing, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
}  //  namespace

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<TestMetavariables>();
  Parallel::charmxx::register_init_node_and_proc({}, {});
}
