// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelSimpleProfilerData.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/SimpleProfilerDataComponent.hpp"
#include "Parallel/Tags/SimpleProfilerData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SimpleProfiler.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct MockSimpleProfilerDataComponent {
  using chare_type = ActionTesting::MockNodeGroupChare;
  using component_being_mocked =
      profiling::SimpleProfilerDataComponent<Metavariables>;

  using simple_tags = tmpl::list<Parallel::Tags::SimpleProfilerData>;

  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct MockArray {
  using chare_type = ActionTesting::MockArrayChare;

  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <bool Enable>
struct test_metavariables {
  using component_list = tmpl::conditional_t<
      Enable,
      tmpl::list<MockSimpleProfilerDataComponent<test_metavariables>,
                 MockArray<test_metavariables>>,
      tmpl::list<MockArray<test_metavariables>>>;
};

struct TestParallelSimpleProfilerData {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const profiling::SimpleProfilerData* const expected_data) {
    CHECK(profiling::parallel_simple_profiler_data(cache) == expected_data);
  }
};

template <bool Enable>
void test() {
  using metavars = test_metavariables<Enable>;
  const size_t cores_per_node = 5;

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {}, {}, {cores_per_node, cores_per_node}};
  for (size_t i = 0; i < cores_per_node; ++i) {
    ActionTesting::emplace_array_component<MockArray<metavars>>(
        &runner, ActionTesting::NodeId{1}, ActionTesting::LocalCoreId{i},
        cores_per_node + i);
  }

  std::vector<const profiling::SimpleProfilerData*> expected(cores_per_node,
                                                             nullptr);
  if constexpr (Enable) {
    ActionTesting::emplace_nodegroup_component_and_initialize<
        MockSimpleProfilerDataComponent<metavars>>(
        &runner, {Parallel::Tags::SimpleProfilerData::type{cores_per_node}});
    const auto& simple_profiler_data = ActionTesting::get_databox_tag<
        MockSimpleProfilerDataComponent<metavars>,
        Parallel::Tags::SimpleProfilerData>(runner, 1);
    for (size_t i = 0; i < cores_per_node; ++i) {
      expected[i] = &simple_profiler_data[i];
    }
  }

  for (size_t i = 0; i < cores_per_node; ++i) {
    ActionTesting::simple_action<MockArray<metavars>,
                                 TestParallelSimpleProfilerData>(
        make_not_null(&runner), cores_per_node + i, expected[i]);
  }
}

SPECTRE_TEST_CASE("Unit.Parallel.ParallelSimpleProfilerData",
                  "[Unit][Parallel]") {
  test<true>();
  test<false>();
}
}  // namespace
