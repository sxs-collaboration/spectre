// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/Matrix.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/MemoryMonitor/Tags.hpp"
#include "Parallel/Serialize.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ContributeMemoryData.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessArray.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessGroups.hpp"
#include "ParallelAlgorithms/Actions/MemoryMonitor/ProcessSingleton.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace {
template <typename Metavariables>
struct MockMemoryMonitor {
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using component_being_mocked = mem_monitor::MemoryMonitor<Metavariables>;
  using metavariables = Metavariables;
  using simple_tags = tmpl::list<mem_monitor::Tags::MemoryHolder>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct SingletonParallelComponent {
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct GroupParallelComponent {
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct NodegroupParallelComponent {
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

template <typename Metavariables>
struct ArrayParallelComponent {
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct TestMetavariables {
  using component_list =
      tmpl::list<MockMemoryMonitor<TestMetavariables>,
                 SingletonParallelComponent<TestMetavariables>,
                 GroupParallelComponent<TestMetavariables>,
                 ArrayParallelComponent<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;

  enum class Phase { Initialization, Monitor, Exit };

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

  enum class Phase { Initialization, Monitor, Exit };

  void pup(PUP::er& /*p*/) {}
};

using metavars = TestMetavarsActions;
using mem_mon_comp = MockMemoryMonitor<metavars>;
using obs_writer_comp = TestHelpers::observers::MockObserverWriter<metavars>;
using sing_comp = SingletonParallelComponent<metavars>;
using group_comp = GroupParallelComponent<metavars>;
using array_comp = ArrayParallelComponent<metavars>;
using nodegroup_comp = NodegroupParallelComponent<metavars>;

void setup_runner(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<metavars>*> runner) {
  // Setup all components even if we aren't using all of them
  ActionTesting::emplace_singleton_component<sing_comp>(
      runner, ActionTesting::NodeId{1}, ActionTesting::LocalCoreId{1});
  ActionTesting::emplace_group_component<group_comp>(runner);
  ActionTesting::emplace_nodegroup_component<nodegroup_comp>(runner);
  ActionTesting::emplace_array_component<array_comp>(
      runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0);

  // ObserverWriter
  ActionTesting::emplace_nodegroup_component_and_initialize<obs_writer_comp>(
      runner, {});

  // MemoryMonitor
  ActionTesting::emplace_singleton_component_and_initialize<mem_mon_comp>(
      runner, ActionTesting::NodeId{2}, ActionTesting::LocalCoreId{2}, {});

  runner->set_phase(metavars::Phase::Monitor);
}

template <typename Component, typename Metavariables>
void check_output(const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
                  const double time, const size_t num_nodes,
                  const std::vector<double>& sizes) {
  auto& read_file = ActionTesting::get_databox_tag<
      obs_writer_comp, TestHelpers::observers::MockReductionFileTag>(runner, 0);
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
      CHECK(data(0, i + 1) == sizes[i]);
    }
  }
  CHECK(data(0, num_columns - 1) == average);
}

template <typename Component>
void run_group_actions(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<metavars>*> runner,
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

  CHECK(ActionTesting::number_of_queued_simple_actions<mem_mon_comp>(
            *runner, 0) == num_branches);

  const auto& mem_holder_tag =
      ActionTesting::get_databox_tag<mem_mon_comp,
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
    ActionTesting::invoke_queued_simple_action<mem_mon_comp>(runner, 0);
  }

  // After we invoke the actions, the map for the current component should be
  // empty because the time should have been erased
  CHECK(mem_holder_tag.at(name).empty());

  // The last action should have called a threaded action to write data
  CHECK(ActionTesting::number_of_queued_threaded_actions<obs_writer_comp>(
            *runner, 0) == 1);
  ActionTesting::invoke_queued_threaded_action<obs_writer_comp>(runner, 0);

  check_output<Component>(*runner, time, static_cast<size_t>(num_nodes), sizes);
}

void test_wrong_box() {
  CHECK_THROWS_WITH(
      ([]() {
        db::DataBox<tmpl::list<>> empty_box{};
        Parallel::GlobalCache<metavars> cache{};

        mem_monitor::ContributeMemoryData<group_comp>::template apply<
            mem_mon_comp>(empty_box, cache, 0, 0.0, 0_st, 0.0);
      }()),
      Catch::Contains("Expected the DataBox for the MemoryMonitor"));
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

  auto& cache = ActionTesting::cache<mem_mon_comp>(runner, 0);
  auto& mem_monitor_proxy =
      Parallel::get_parallel_component<mem_mon_comp>(cache);

  const double time = 0.5;
  std::vector<double> sizes(num_nodes);
  size_t index = 0;
  // Do the group first. If we are using the ProcessGroup action, invoke all the
  // simple actions on the group and calculate the size of that component so we
  // can compare, otherwise call the ContributeMemoryData action directly with a
  // random size
  if (use_process_component_actions) {
    auto& group_proxy = Parallel::get_parallel_component<group_comp>(cache);
    Parallel::simple_action<mem_monitor::ProcessGroups>(group_proxy, time);

    for (size_t proc = 0; proc < num_procs; proc++) {
      CHECK(ActionTesting::number_of_queued_simple_actions<group_comp>(
                runner, proc) == 1);
      if (proc != 0 and proc % num_procs_per_node == 0) {
        ++index;
      }
      sizes[index] +=
          size_of_object_in_bytes(*Parallel::local_branch(group_proxy)) / 1.0e6;
      ActionTesting::invoke_queued_simple_action<group_comp>(
          make_not_null(&runner), proc);
    }
  } else {
    for (size_t proc = 0; proc < num_procs; proc++) {
      const double size = dist(*gen);
      if (proc != 0 and proc % num_procs_per_node == 0) {
        ++index;
      }
      sizes[index] += size;
      Parallel::simple_action<mem_monitor::ContributeMemoryData<group_comp>>(
          mem_monitor_proxy, time, static_cast<int>(proc), size);
    }
  }

  run_group_actions<group_comp>(make_not_null(&runner),
                                static_cast<int>(num_nodes),
                                static_cast<int>(num_procs), time, sizes);

  sizes.clear();
  sizes.resize(num_nodes);

  // Now for the nodegroup
  if (use_process_component_actions) {
    auto& nodegroup_proxy =
        Parallel::get_parallel_component<nodegroup_comp>(cache);
    Parallel::simple_action<mem_monitor::ProcessGroups>(nodegroup_proxy, time);

    for (size_t node = 0; node < num_nodes; node++) {
      CHECK(ActionTesting::number_of_queued_simple_actions<nodegroup_comp>(
                runner, node) == 1);
      sizes[node] =
          size_of_object_in_bytes(*Parallel::local_branch(nodegroup_proxy)) /
          1.0e6;
      ActionTesting::invoke_queued_simple_action<nodegroup_comp>(
          make_not_null(&runner), node);
    }
  } else {
    for (size_t node = 0; node < num_nodes; node++) {
      const double size = dist(*gen);
      sizes[node] = size;
      Parallel::simple_action<
          mem_monitor::ContributeMemoryData<nodegroup_comp>>(
          mem_monitor_proxy, time, static_cast<int>(node), size);
    }
  }

  run_group_actions<nodegroup_comp>(make_not_null(&runner),
                                    static_cast<int>(num_nodes),
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

  auto& cache = ActionTesting::cache<mem_mon_comp>(runner, 0);
  auto& mem_monitor_proxy =
      Parallel::get_parallel_component<mem_mon_comp>(cache);

  const double time = 0.5;
  std::vector<double> size_per_node(num_nodes);
  fill_with_random_values(make_not_null(&size_per_node), gen,
                          make_not_null(&dist));

  Parallel::simple_action<mem_monitor::ProcessArray<array_comp>>(
      mem_monitor_proxy, time, size_per_node);
  CHECK(ActionTesting::number_of_queued_simple_actions<mem_mon_comp>(runner,
                                                                     0) == 1);
  ActionTesting::invoke_queued_simple_action<mem_mon_comp>(
      make_not_null(&runner), 0);

  CHECK(ActionTesting::number_of_queued_threaded_actions<obs_writer_comp>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_threaded_action<obs_writer_comp>(
      make_not_null(&runner), 0);

  check_output<array_comp>(runner, time, num_nodes, size_per_node);
}

void test_process_singleton() {
  INFO("Test ProcessSingleton");

  // 4 mock nodes, 3 mock cores per node
  const size_t num_nodes = 4;
  const size_t num_procs_per_node = 3;
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {}, {}, std::vector<size_t>(num_nodes, num_procs_per_node)};

  setup_runner(make_not_null(&runner));

  auto& cache = ActionTesting::cache<sing_comp>(runner, 0);
  auto& singleton_proxy = Parallel::get_parallel_component<sing_comp>(cache);

  const double time = 0.5;
  std::vector<double> sizes(num_nodes, 0.0);

  Parallel::simple_action<mem_monitor::ProcessSingleton>(singleton_proxy, time);
  CHECK(ActionTesting::number_of_queued_simple_actions<sing_comp>(runner, 0) ==
        1);

  // We multiply by the number of nodes here because in the checK_output()
  // function, it takes an average over number of nodes to accommodate arrays
  // and (node)groups. But for singletons, we don't need an average because it's
  // just one measurement. So we "undo" the average here by multiplying by the
  // number of nodes
  sizes[0] = static_cast<double>(num_nodes) *
             size_of_object_in_bytes(*Parallel::local(singleton_proxy)) / 1.0e6;

  ActionTesting::invoke_queued_simple_action<sing_comp>(make_not_null(&runner),
                                                        0);

  CHECK(ActionTesting::number_of_queued_threaded_actions<obs_writer_comp>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_threaded_action<obs_writer_comp>(
      make_not_null(&runner), 0);

  check_output<sing_comp>(runner, time, num_nodes, sizes);
}

SPECTRE_TEST_CASE("Unit.Parallel.MemoryMonitor", "[Unit][Parallel]") {
  MAKE_GENERATOR(gen);
  test_tags();
  test_wrong_box();
  // First only test the ContributeMemoryData action (second arg false)
  test_contribute_memory_data(make_not_null(&gen), false);
  // Then test the Process(Node)Group actions (second arg true)
  test_contribute_memory_data(make_not_null(&gen), true);
  test_process_array(make_not_null(&gen));
  test_process_singleton();
}
}  // namespace
