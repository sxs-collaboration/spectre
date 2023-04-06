// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/MemoryMonitor/Tags.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ContributeMemoryData.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessArray.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessGroups.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessSingleton.hpp"
#include "ParallelAlgorithms/Events/MonitorMemory.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace {
struct TimeTag {
  using type = double;
};

template <typename Metavariables>
struct MockMemoryMonitor {
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using component_being_mocked = mem_monitor::MemoryMonitor<Metavariables>;
  using metavariables = Metavariables;
  using simple_tags = tmpl::list<mem_monitor::Tags::MemoryHolder>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct SingletonParallelComponent {
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct GroupParallelComponent {
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct NodegroupParallelComponent {
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct ArrayParallelComponent {
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

// This component deserves special mention. It is supposed to be playing the
// role of the DgElementArray, the component where we run the MonitorMemory
// event. However, inside MonitorMemory, there is an `if constexpr` check if the
// component we are monitoring is an array. If we are monitoring an array, then
// Parallel::contribute_to_reduction is called. If we make this component a
// MockArray, the test fails to build because the ATF doesn't support
// reductions yet. To get around this, we make this component a MockSingleton
// because the event only needs to be run on one "element". We also remove it
// from the components to monitor. Once the ATF supports reductions, this can be
// changed to a MockArray.
template <typename Metavariables>
struct FakeDgElementArray {
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 tmpl::list<domain::Tags::Element<3>>>>>>;
};

struct TestMetavariables {
  using component_list =
      tmpl::list<MockMemoryMonitor<TestMetavariables>,
                 SingletonParallelComponent<TestMetavariables>,
                 TestHelpers::observers::MockObserverWriter<TestMetavariables>,
                 GroupParallelComponent<TestMetavariables>,
                 FakeDgElementArray<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;


  void pup(PUP::er& /*p*/) {}
};

void test_tags() {
  INFO("Test Tags");
  using holder_tag = mem_monitor::Tags::MemoryHolder;
  TestHelpers::db::test_simple_tag<holder_tag>("MemoryHolder");

  const std::string subpath =
      mem_monitor::subfile_name<MockMemoryMonitor<TestMetavariables>>();

  CHECK(subpath == "/MemoryMonitors/MockMemoryMonitor");
}

struct TestMetavarsActions {
  using observed_reduction_data_tags = tmpl::list<>;

  using component_list = tmpl::list<
      MockMemoryMonitor<TestMetavarsActions>,
      TestHelpers::observers::MockObserverWriter<TestMetavarsActions>,
      SingletonParallelComponent<TestMetavarsActions>,
      GroupParallelComponent<TestMetavarsActions>,
      ArrayParallelComponent<TestMetavarsActions>,
      NodegroupParallelComponent<TestMetavarsActions>>;


  void pup(PUP::er& /*p*/) {}
};

using metavars = TestMetavarsActions;
template <typename Metavars>
using mem_mon_comp = MockMemoryMonitor<Metavars>;
template <typename Metavars>
using obs_writer_comp = TestHelpers::observers::MockObserverWriter<Metavars>;
template <typename Metavars>
using sing_comp = SingletonParallelComponent<Metavars>;
template <typename Metavars>
using group_comp = GroupParallelComponent<Metavars>;
template <typename Metavars>
using array_comp = ArrayParallelComponent<Metavars>;
template <typename Metavars>
using nodegroup_comp = NodegroupParallelComponent<Metavars>;
template <typename Metavars>
using dg_elem_comp = FakeDgElementArray<Metavars>;

template <typename Metavariables>
void setup_runner(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavariables>*>
        runner) {
  // Setup all components even if we aren't using all of them
  ActionTesting::emplace_singleton_component<sing_comp<Metavariables>>(
      runner, ActionTesting::NodeId{1}, ActionTesting::LocalCoreId{1});
  ActionTesting::emplace_group_component<group_comp<Metavariables>>(runner);
  ActionTesting::emplace_nodegroup_component<nodegroup_comp<Metavariables>>(
      runner);
  if constexpr (tmpl::list_contains_v<typename Metavariables::component_list,
                                      array_comp<Metavariables>>) {
    ActionTesting::emplace_array_component<array_comp<Metavariables>>(
        runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0);
  }

  // ObserverWriter
  ActionTesting::emplace_nodegroup_component_and_initialize<
      obs_writer_comp<Metavariables>>(runner, {});

  // MemoryMonitor
  ActionTesting::emplace_singleton_component_and_initialize<
      mem_mon_comp<Metavariables>>(runner, ActionTesting::NodeId{2},
                                   ActionTesting::LocalCoreId{2}, {});

  if constexpr (tmpl::list_contains_v<typename Metavariables::component_list,
                                      dg_elem_comp<Metavariables>>) {
    // FakeDgElementArray that's actually a singleton
    const ElementId<3> element_id{0};
    const Element<3> element{element_id, {}};
    ActionTesting::emplace_singleton_component_and_initialize<
        dg_elem_comp<Metavariables>>(runner, ActionTesting::NodeId{0},
                                     ActionTesting::LocalCoreId{0}, {element});
  }

  runner->set_phase(Parallel::Phase::Testing);
}

template <typename Component, typename Metavariables>
void check_output(const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
                  const double time, const size_t num_nodes,
                  const std::vector<double>& sizes) {
  auto& read_file = ActionTesting::get_databox_tag<
      obs_writer_comp<Metavariables>,
      TestHelpers::observers::MockReductionFileTag>(runner, 0);
  INFO("Checking output of " + pretty_type::name<Component>());
  const auto& dataset =
      read_file.get_dat(mem_monitor::subfile_name<Component>());
  const std::vector<std::string>& legend = dataset.get_legend();

  size_t num_columns;
  if constexpr (Parallel::is_singleton_v<Component>) {
    // time, proc, size
    num_columns = 3;
  } else if constexpr (Parallel::is_group_v<Component>) {
    // time, size on node 0, size on node 1, ...,  proc of max size, max size,
    // avg per node
    num_columns = num_nodes + 4;
  } else {
    // time, size on node 0, size on node 1, ..., avg per node
    num_columns = num_nodes + 2;
  }

  CHECK(legend.size() == num_columns);

  const Matrix data = dataset.get_data();
  // We only wrote one line of data for all components
  CHECK(data.rows() == 1);
  CHECK(data.columns() == num_columns);

  const double average =
      alg::accumulate(sizes, 0.0) / static_cast<double>(num_nodes);

  CHECK(data(0, 0) == time);
  // Singletons don't report sizes on each node because there is only one
  // measurement
  if constexpr (not Parallel::is_singleton_v<Component>) {
    for (size_t i = 0; i < num_nodes; i++) {
      INFO("i = " + get_output(i));
      CHECK(data(0, i + 1) == sizes[i]);
    }
  }
  CHECK(data(0, num_columns - 1) == average);
}

template <typename Component, typename Metavariables>
void run_group_actions(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavariables>*>
        runner,
    const int num_nodes, const int num_procs, const double time,
    const std::vector<double>& sizes) {
  INFO("Checking actions of " + pretty_type::name<Component>());
  size_t num_branches;
  if constexpr (Parallel::is_group_v<Component>) {
    num_branches = static_cast<size_t>(num_procs);
  } else {
    num_branches = static_cast<size_t>(num_nodes);
    // Avoid unused parameter warning
    (void)num_procs;
  }

  CHECK(ActionTesting::number_of_queued_simple_actions<
            mem_mon_comp<Metavariables>>(*runner, 0) == num_branches);

  const auto& mem_holder_tag =
      ActionTesting::get_databox_tag<mem_mon_comp<Metavariables>,
                                     mem_monitor::Tags::MemoryHolder>(*runner,
                                                                      0);

  // Before we invoke the actions, the map for this component shouldn't exist
  // because this is the first time we are calling it.
  const std::string name = pretty_type::name<Component>();
  CHECK(mem_holder_tag.count(name) == 0);

  for (size_t i = 0; i < num_branches; i++) {
    // unordered_map<name, unordered_map<time, unordered_map<node/proc, size>>>
    // Putting the CHECK before the action invocation is arbitrary. The map at a
    // specific time is empty both before the first action is invoked and after
    // the last action is invoked. Because of this, we need to put the CHECK in
    // an if statement to avoid an access error, whether the check happens
    // before or after the invocation.
    if (i != 0) {
      CHECK(mem_holder_tag.at(name).at(time).size() == i);
    }
    ActionTesting::invoke_queued_simple_action<mem_mon_comp<Metavariables>>(
        runner, 0);
  }

  // After we invoke the actions, the map for the current component should be
  // empty because the time should have been erased
  CHECK(mem_holder_tag.at(name).empty());

  // The last action should have called a threaded action to write data
  CHECK(ActionTesting::number_of_queued_threaded_actions<
            obs_writer_comp<Metavariables>>(*runner, 0) == 1);
  ActionTesting::invoke_queued_threaded_action<obs_writer_comp<Metavariables>>(
      runner, 0);

  check_output<Component>(*runner, time, static_cast<size_t>(num_nodes), sizes);
}

// Contribute memory data can only be used with groups or nodegroups
template <typename Gen>
void test_contribute_memory_data(const gsl::not_null<Gen*> gen,
                                 const bool use_process_component_actions) {
  INFO("Test ContributeMemoryData with" +
       (use_process_component_actions ? ""s : "out"s) + " Process(Node)Group");
  std::uniform_real_distribution<double> dist{0, 10.0};

  // 4 mock nodes, 3 mock cores per node
  const size_t num_nodes = 4;
  const size_t num_procs_per_node = 3;
  const size_t num_procs = num_nodes * num_procs_per_node;
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {}, {}, std::vector<size_t>(num_nodes, num_procs_per_node)};

  setup_runner(make_not_null(&runner));

  auto& cache = ActionTesting::cache<mem_mon_comp<metavars>>(runner, 0);
  auto& mem_monitor_proxy =
      Parallel::get_parallel_component<mem_mon_comp<metavars>>(cache);

  const double time = 0.5;
  std::vector<double> sizes(num_nodes);
  size_t index = 0;
  // Do the group first. If we are using the ProcessGroup action, invoke all the
  // simple actions on the group and calculate the size of that component so we
  // can compare, otherwise call the ContributeMemoryData action directly with a
  // random size
  if (use_process_component_actions) {
    auto& group_proxy =
        Parallel::get_parallel_component<group_comp<metavars>>(cache);
    Parallel::simple_action<mem_monitor::ProcessGroups>(group_proxy, time);

    for (size_t proc = 0; proc < num_procs; proc++) {
      CHECK(
          ActionTesting::number_of_queued_simple_actions<group_comp<metavars>>(
              runner, proc) == 1);
      if (proc != 0 and proc % num_procs_per_node == 0) {
        ++index;
      }
      sizes[index] +=
          size_of_object_in_bytes(*Parallel::local_branch(group_proxy)) / 1.0e6;
      ActionTesting::invoke_queued_simple_action<group_comp<metavars>>(
          make_not_null(&runner), proc);
    }
  } else {
    for (size_t proc = 0; proc < num_procs; proc++) {
      const double size = dist(*gen);
      if (proc != 0 and proc % num_procs_per_node == 0) {
        ++index;
      }
      sizes[index] += size;
      Parallel::simple_action<
          mem_monitor::ContributeMemoryData<group_comp<metavars>>>(
          mem_monitor_proxy, time, static_cast<int>(proc), size);
    }
  }

  run_group_actions<group_comp<metavars>>(
      make_not_null(&runner), static_cast<int>(num_nodes),
      static_cast<int>(num_procs), time, sizes);

  sizes.clear();
  sizes.resize(num_nodes);

  // Now for the nodegroup
  if (use_process_component_actions) {
    auto& nodegroup_proxy =
        Parallel::get_parallel_component<nodegroup_comp<metavars>>(cache);
    Parallel::simple_action<mem_monitor::ProcessGroups>(nodegroup_proxy, time);

    for (size_t node = 0; node < num_nodes; node++) {
      CHECK(ActionTesting::number_of_queued_simple_actions<
                nodegroup_comp<metavars>>(runner, node) == 1);
      sizes[node] =
          size_of_object_in_bytes(*Parallel::local_branch(nodegroup_proxy)) /
          1.0e6;
      ActionTesting::invoke_queued_simple_action<nodegroup_comp<metavars>>(
          make_not_null(&runner), node);
    }
  } else {
    for (size_t node = 0; node < num_nodes; node++) {
      const double size = dist(*gen);
      sizes[node] = size;
      Parallel::simple_action<
          mem_monitor::ContributeMemoryData<nodegroup_comp<metavars>>>(
          mem_monitor_proxy, time, static_cast<int>(node), size);
    }
  }

  run_group_actions<nodegroup_comp<metavars>>(
      make_not_null(&runner), static_cast<int>(num_nodes),
      static_cast<int>(num_procs), time, sizes);
}

template <typename Gen>
void test_process_array(const gsl::not_null<Gen*> gen) {
  INFO("Test ProcessArray");
  std::uniform_real_distribution<double> dist{0, 10.0};

  // 4 mock nodes, 3 mock cores per node
  const size_t num_nodes = 4;
  const size_t num_procs_per_node = 3;
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {}, {}, std::vector<size_t>(num_nodes, num_procs_per_node)};

  setup_runner(make_not_null(&runner));

  auto& cache = ActionTesting::cache<mem_mon_comp<metavars>>(runner, 0);
  auto& mem_monitor_proxy =
      Parallel::get_parallel_component<mem_mon_comp<metavars>>(cache);

  const double time = 0.5;
  std::vector<double> size_per_node(num_nodes);
  fill_with_random_values(make_not_null(&size_per_node), gen,
                          make_not_null(&dist));

  Parallel::simple_action<mem_monitor::ProcessArray<array_comp<metavars>>>(
      mem_monitor_proxy, time, size_per_node);
  CHECK(ActionTesting::number_of_queued_simple_actions<mem_mon_comp<metavars>>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_simple_action<mem_mon_comp<metavars>>(
      make_not_null(&runner), 0);

  CHECK(ActionTesting::number_of_queued_threaded_actions<
            obs_writer_comp<metavars>>(runner, 0) == 1);
  ActionTesting::invoke_queued_threaded_action<obs_writer_comp<metavars>>(
      make_not_null(&runner), 0);

  check_output<array_comp<metavars>>(runner, time, num_nodes, size_per_node);
}

void test_process_singleton() {
  INFO("Test ProcessSingleton");

  // 4 mock nodes, 3 mock cores per node
  const size_t num_nodes = 4;
  const size_t num_procs_per_node = 3;
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {}, {}, std::vector<size_t>(num_nodes, num_procs_per_node)};

  setup_runner(make_not_null(&runner));

  auto& cache = ActionTesting::cache<sing_comp<metavars>>(runner, 0);
  auto& singleton_proxy =
      Parallel::get_parallel_component<sing_comp<metavars>>(cache);

  const double time = 0.5;
  std::vector<double> sizes(num_nodes, 0.0);

  Parallel::simple_action<mem_monitor::ProcessSingleton>(singleton_proxy, time);
  CHECK(ActionTesting::number_of_queued_simple_actions<sing_comp<metavars>>(
            runner, 0) == 1);

  // We multiply by the number of nodes here because in the checK_output()
  // function, it takes an average over number of nodes to accommodate arrays
  // and (node)groups. But for singletons, we don't need an average because it's
  // just one measurement. So we "undo" the average here by multiplying by the
  // number of nodes
  sizes[0] = static_cast<double>(num_nodes) *
             size_of_object_in_bytes(*Parallel::local(singleton_proxy)) / 1.0e6;

  ActionTesting::invoke_queued_simple_action<sing_comp<metavars>>(
      make_not_null(&runner), 0);

  CHECK(ActionTesting::number_of_queued_threaded_actions<
            obs_writer_comp<metavars>>(runner, 0) == 1);
  ActionTesting::invoke_queued_threaded_action<obs_writer_comp<metavars>>(
      make_not_null(&runner), 0);

  check_output<sing_comp<metavars>>(runner, time, num_nodes, sizes);
}

struct BadArrayChareMetavariables {
  using component_list =
      tmpl::list<ArrayParallelComponent<BadArrayChareMetavariables>>;

};

void test_event_construction() {
  CHECK_THROWS_WITH(
      ([]() {
        std::vector<std::string> misspelled_component{"GlabolCahce"};

        Events::MonitorMemory<1, TimeTag> event{
            {misspelled_component}, Options::Context{}, metavars{}};
      }()),
      Catch::Contains(
          "Cannot monitor memory usage of unknown parallel component"));

  CHECK_THROWS_WITH(
      ([]() {
        std::vector<std::string> array_component{"ArrayParallelComponent"};

        Events::MonitorMemory<2, TimeTag> event{{array_component},
                                                Options::Context{},
                                                BadArrayChareMetavariables{}};
      }()),
      Catch::Contains("Currently, the only Array parallel component allowed to "
                      "be monitored is the DgElementArray."));
}

void test_monitor_memory_event() {
  using event_metavars = TestMetavariables;
  INFO("Checking MonitorMemory event");

  // 4 mock nodes, 3 mock cores per node
  const size_t num_nodes = 4;
  const size_t num_procs_per_node = 3;
  const size_t num_procs = num_nodes * num_procs_per_node;
  ActionTesting::MockRuntimeSystem<event_metavars> runner{
      {}, {}, std::vector<size_t>(num_nodes, num_procs_per_node)};

  setup_runner(make_not_null(&runner));

  auto& cache = ActionTesting::cache<mem_mon_comp<event_metavars>>(runner, 0);

  std::vector<std::string> components_to_monitor{};
  // Note that we don't monitor the global caches here. This is because the
  // entry methods of the global caches used to compute memory and send it to
  // the MemoryMonitor are not compatible with the ATF. Also don't monitor the
  // DgElementArray or the array component (because we can't in the ATF)
  using component_list =
      tmpl::list_difference<typename event_metavars::component_list,
                            tmpl::list<dg_elem_comp<event_metavars>>>;
  tmpl::for_each<component_list>([&components_to_monitor](auto component_v) {
    using component = tmpl::type_from<decltype(component_v)>;
    components_to_monitor.emplace_back(pretty_type::name<component>());
  });

  // Create event
  Events::MonitorMemory<3, TimeTag> monitor_memory{
      {components_to_monitor}, Options::Context{}, metavars{}};

  const auto& element =
      ActionTesting::get_databox_tag<dg_elem_comp<event_metavars>,
                                     domain::Tags::Element<3>>(runner, 0);

  // Run the event. This will queue a lot of actions
  const double time = 1.4;
  monitor_memory(time, element, cache, 0,
                 std::add_pointer_t<dg_elem_comp<event_metavars>>{});

  // Check how many simple actions are queued:
  // - MemoryMonitor: 1
  // - ObserverWriter: 4 (one per node)
  // - Singleton: 1
  // - Group: 12 (one per core)
  // - NodeGroup: 4 (one per node)
  tmpl::for_each<component_list>([&runner](auto component_v) {
    using component = tmpl::type_from<decltype(component_v)>;
    if constexpr (Parallel::is_singleton_v<component>) {
      CHECK(ActionTesting::number_of_queued_simple_actions<component>(runner,
                                                                      0) == 1);
    } else if constexpr (Parallel::is_group_v<component>) {
      for (int i = 0; i < static_cast<int>(num_procs); i++) {
        CHECK(ActionTesting::number_of_queued_simple_actions<component>(
                  runner, i) == 1);
      }
    } else if constexpr (Parallel::is_nodegroup_v<component>) {
      for (int i = 0; i < static_cast<int>(num_nodes); i++) {
        CHECK(ActionTesting::number_of_queued_simple_actions<component>(
                  runner, i) == 1);
      }
    }
  });

  // First invoke the simple actions on the singletons.
  ActionTesting::invoke_queued_simple_action<mem_mon_comp<event_metavars>>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_simple_action<sing_comp<event_metavars>>(
      make_not_null(&runner), 0);

  // These immediately call the observer writer to write data to disk
  CHECK(ActionTesting::number_of_queued_threaded_actions<
            obs_writer_comp<event_metavars>>(runner, 0) == 2);

  ActionTesting::invoke_queued_threaded_action<obs_writer_comp<event_metavars>>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer_comp<event_metavars>>(
      make_not_null(&runner), 0);

  // Check that the data was written correctly
  std::vector<double> sing_sizes(num_nodes, 0.0);
  // We multiply by the number of nodes for same reason as in
  // test_process_singleton()
  auto& mem_mon_proxy =
      Parallel::get_parallel_component<mem_mon_comp<event_metavars>>(cache);
  sing_sizes[0] = static_cast<double>(num_nodes) *
                  size_of_object_in_bytes(*Parallel::local(mem_mon_proxy)) /
                  1.0e6;
  check_output<mem_mon_comp<event_metavars>>(runner, time, num_nodes,
                                             sing_sizes);

  auto& sing_cache = ActionTesting::cache<sing_comp<event_metavars>>(runner, 0);
  auto& sing_proxy =
      Parallel::get_parallel_component<sing_comp<event_metavars>>(sing_cache);
  sing_sizes[0] = static_cast<double>(num_nodes) *
                  size_of_object_in_bytes(*Parallel::local(sing_proxy)) / 1.0e6;
  check_output<sing_comp<event_metavars>>(runner, time, num_nodes, sing_sizes);

  // Now for the groups and nodegroups
  using group_list =
      tmpl::list<obs_writer_comp<event_metavars>, group_comp<event_metavars>,
                 nodegroup_comp<event_metavars>>;
  tmpl::for_each<group_list>([&runner, &num_nodes, &num_procs, &time,
                              &cache](auto component_v) {
    using component = tmpl::type_from<decltype(component_v)>;
    std::vector<double> sizes(num_nodes);
    auto& proxy = Parallel::get_parallel_component<component>(cache);
    Parallel::simple_action<mem_monitor::ProcessGroups>(proxy, time);

    for (size_t node = 0; node < num_nodes; node++) {
      if constexpr (Parallel::is_nodegroup_v<component>) {
        // Need the cache of this specific component to get the proper size
        auto& local_cache = ActionTesting::cache<component>(runner, node);
        auto& local_proxy =
            Parallel::get_parallel_component<component>(local_cache);
        sizes[node] =
            size_of_object_in_bytes(*Parallel::local_branch(local_proxy)) /
            1.0e6;
        ActionTesting::invoke_queued_simple_action<component>(
            make_not_null(&runner), node);

      } else {
        for (size_t proc = 0; proc < num_procs_per_node; proc++) {
          const size_t global_proc = node * num_procs_per_node + proc;
          // Need the local cache to get the proper size
          auto& local_cache =
              ActionTesting::cache<component>(runner, global_proc);
          auto& local_proxy =
              Parallel::get_parallel_component<component>(local_cache);
          sizes[node] +=
              size_of_object_in_bytes(*Parallel::local_branch(local_proxy)) /
              1.0e6;
          ActionTesting::invoke_queued_simple_action<component>(
              make_not_null(&runner), global_proc);
        }
      }
    }

    run_group_actions<component>(make_not_null(&runner), num_nodes, num_procs,
                                 time, sizes);
  });

  // All actions should be completed now
  CHECK(ActionTesting::number_of_queued_simple_actions<
            mem_mon_comp<event_metavars>>(runner, 0) == 0);
  CHECK(ActionTesting::number_of_queued_threaded_actions<
            obs_writer_comp<event_metavars>>(runner, 0) == 0);
}

SPECTRE_TEST_CASE("Unit.Parallel.MemoryMonitor", "[Unit][Parallel]") {
  MAKE_GENERATOR(gen);
  test_tags();
  // First only test the ContributeMemoryData action (second arg false)
  test_contribute_memory_data(make_not_null(&gen), false);
  // Then test the Process(Node)Group actions (second arg true)
  test_contribute_memory_data(make_not_null(&gen), true);
  test_process_array(make_not_null(&gen));
  test_process_singleton();
  test_event_construction();
  test_monitor_memory_event();
}
}  // namespace
