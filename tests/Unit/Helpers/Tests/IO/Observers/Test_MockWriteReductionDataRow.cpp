// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/MockH5.hpp"
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::observers {
namespace {
struct TestMetavariables;

using mock_observer_writer = MockObserverWriter<TestMetavariables>;
struct TestMetavariables {
  using component_list = tmpl::list<mock_observer_writer>;

};

void run_test() {
  ActionTesting::MockRuntimeSystem<TestMetavariables> runner{{}};

  // [initialize_component]
  ActionTesting::emplace_nodegroup_component_and_initialize<
      mock_observer_writer>(make_not_null(&runner), {});
  // [initialize_component]

  REQUIRE(ActionTesting::tag_is_retrievable<mock_observer_writer,
                                            MockReductionFileTag>(runner, 0));

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& cache = ActionTesting::cache<mock_observer_writer>(runner, 0);

  auto& proxy = Parallel::get_parallel_component<mock_observer_writer>(cache);

  const std::string subfile_path{"/path/to/something"};
  const std::vector<std::string> legend{{"Time"s, "Value"s}};

  Parallel::threaded_action<
      ::observers::ThreadedActions::WriteReductionDataRow>(
      proxy[0], subfile_path, legend, std::make_tuple(1.0, 9.3));

  ActionTesting::invoke_queued_threaded_action<mock_observer_writer>(
      make_not_null(&runner), 0);

  // [check_mock_writer_data]
  auto& mock_h5_file =
      ActionTesting::get_databox_tag<mock_observer_writer,
                                     MockReductionFileTag>(runner, 0);

  const auto& mock_dat_file = mock_h5_file.get_dat(subfile_path);
  CHECK(mock_dat_file.get_legend() == legend);
  CHECK(mock_dat_file.get_data() == Matrix{{1.0, 9.3}});
  // [check_mock_writer_data]
}

SPECTRE_TEST_CASE("Test.TestHelpers.IO.Observers.MockWriteReductionDataRow",
                  "[IO][Unit]") {
  run_test();
}
}  // namespace
}  // namespace TestHelpers::observers
