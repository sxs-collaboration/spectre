// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/IO/Observers/MockH5.hpp"
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/LastSimpleProfilerWrite.hpp"
#include "Parallel/Tags/SimpleProfilerData.hpp"
#include "ParallelAlgorithms/Events/ObserveSimpleProfiler.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace profiling {
template <typename Metavariables>
struct SimpleProfilerDataComponent;
}  // namespace profiling

namespace {
struct TestLabel {
  using FileLabel = int;
  using ColumnLabel = int;
  static constexpr std::array allowed_column_labels{1, 2};

  static std::string make_file_name(const int file_label) {
    return "Test/Label" + std::to_string(file_label);
  }

  static std::string make_column_name(const int column_label) {
    return std::to_string(column_label);
  }
};

template <typename Metavariables>
struct MockSimpleProfilerDataComponent {
  using chare_type = ActionTesting::MockNodeGroupChare;
  using component_being_mocked =
      profiling::SimpleProfilerDataComponent<Metavariables>;

  using simple_tags = tmpl::list<Parallel::Tags::SimpleProfilerData,
                                 Parallel::Tags::LastSimpleProfilerWrite>;

  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct TestMetavars {
  using component_list =
      tmpl::list<MockSimpleProfilerDataComponent<TestMetavars>,
                 ::TestHelpers::observers::MockObserverWriter<TestMetavars>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<Events::ObserveSimpleProfiler>>>;
  };
};

void wait() {
  const timespec spec{.tv_sec = 0, .tv_nsec = 10'000};
  nanosleep(&spec, nullptr);
}

void check_unprofiled(const TestHelpers::observers::MockH5File& file,
                      const size_t proc,
                      const std::vector<double>& expected_ids) {
  const auto& dat =
      file.get_dat("/ProfileTest/Proc" + std::to_string(proc) + "/Unprofiled");
  CHECK(dat.get_legend() ==
        std::vector<std::string>{"ObsName", "count", "exclusive", "inclusive"});
  const auto& data = dat.get_data();
  REQUIRE(data.rows() == expected_ids.size());
  for (size_t i = 0; i < data.rows(); ++i) {
    CHECK(data(i, 0) == expected_ids[i]);
    CHECK(data(i, 1) > 0.0);
    CHECK(data(i, 2) == data(i, 3));
  }
}

enum class Expected : std::uint8_t { Zero, Equal, Unequal };

void check_test_label(
    const TestHelpers::observers::MockH5File& file, const size_t proc,
    const int file_label,
    const std::map<double, std::array<Expected, 2>>& expected) {
  const auto& dat = file.get_dat("/ProfileTest/Proc" + std::to_string(proc) +
                                 "/Test/Label" + std::to_string(file_label));
  CHECK(dat.get_legend() ==
        std::vector<std::string>{"ObsName", "1 count", "1 exclusive",
                                 "1 inclusive", "2 count", "2 exclusive",
                                 "2 inclusive"});
  const auto& data = dat.get_data();
  REQUIRE(data.rows() == expected.size());
  for (size_t i = 0; i < data.rows(); ++i) {
    const auto& expected_row =
        *std::next(expected.begin(), static_cast<std::ptrdiff_t>(i));
    CHECK(data(i, 0) == expected_row.first);
    for (size_t cols = 0; cols < 2; ++cols) {
      switch (gsl::at(expected_row.second, cols)) {
        case Expected::Equal:
          CHECK(data(i, 3 * cols + 1) > 0.0);
          CHECK(data(i, 3 * cols + 2) == data(i, 3 * cols + 3));
          CHECK(data(i, 3 * cols + 2) > 0.0);
          break;
        case Expected::Unequal:
          CHECK(data(i, 3 * cols + 1) > 0.0);
          CHECK(data(i, 3 * cols + 2) < data(i, 3 * cols + 3));
          break;
        default:
          CHECK(data(i, 3 * cols + 1) == 0.0);
          CHECK(data(i, 3 * cols + 2) == 0.0);
          CHECK(data(i, 3 * cols + 3) == 0.0);
          break;
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.ObserveSimpleProfiler",
                  "[Unit][ParallelAlgorithms]") {
  using profiler_component = MockSimpleProfilerDataComponent<TestMetavars>;
  using observer = ::TestHelpers::observers::MockObserverWriter<TestMetavars>;

  const size_t cores_per_node = 3;

  ActionTesting::MockRuntimeSystem<TestMetavars> runner{
      {}, {}, {cores_per_node, cores_per_node}};
  ActionTesting::emplace_nodegroup_component<profiler_component>(&runner);
  ActionTesting::emplace_nodegroup_component<observer>(make_not_null(&runner));

  const std::array profiler_boxes{
      &ActionTesting::get_databox<profiler_component>(make_not_null(&runner),
                                                      0),
      &ActionTesting::get_databox<profiler_component>(make_not_null(&runner),
                                                      1)};
  const std::array profiler_data{
      &db::get_mutable_reference<Parallel::Tags::SimpleProfilerData>(
          make_not_null(profiler_boxes[0])),
      &db::get_mutable_reference<Parallel::Tags::SimpleProfilerData>(
          make_not_null(profiler_boxes[1]))};
  const std::array last_write{
      &db::get_mutable_reference<Parallel::Tags::LastSimpleProfilerWrite>(
          make_not_null(profiler_boxes[0])),
      &db::get_mutable_reference<Parallel::Tags::LastSimpleProfilerWrite>(
          make_not_null(profiler_boxes[1]))};

  for (size_t node = 0; node < 2; ++node) {
    gsl::at(profiler_data, node)->resize(cores_per_node);
    gsl::at(last_write, node)
        ->store(-std::numeric_limits<double>::infinity(),
                std::memory_order_release);
  }

  const auto& file = ActionTesting::get_databox_tag<
      observer, ::TestHelpers::observers::MockReductionFileTag>(runner, 0);

  const auto observe =
      TestHelpers::test_factory_creation<Event, Events::ObserveSimpleProfiler>(
          "ObserveSimpleProfiler:\n"
          "  SubfileName: ProfileTest");
  const auto do_observe = [&last_write, &observe,
                           &runner](const double obs_value) {
    const Event::ObservationValue obs{.name = "ObsName", .value = obs_value};
    db::DataBox<tmpl::list<>> empty_box{};
    auto obs_box =
        make_observation_box<tmpl::list<>>(make_not_null(&empty_box));
    const void* const component_pointer{};

    for (size_t node = 0; node < 2; ++node) {
      observe->run(make_not_null(&obs_box),
                   ActionTesting::cache<observer>(runner, node), 0,
                   component_pointer, obs);
      CHECK(gsl::at(last_write, node)->load(std::memory_order_acquire) ==
            obs_value);
      // Running again does nothing
      observe->run(make_not_null(&obs_box),
                   ActionTesting::cache<observer>(runner, node), 0,
                   component_pointer, obs);
      CHECK(gsl::at(last_write, node)->load(std::memory_order_acquire) ==
            obs_value);
    }

    REQUIRE(ActionTesting::number_of_queued_threaded_actions<observer>(runner,
                                                                       0) == 2);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::invoke_queued_threaded_action<observer>(
          make_not_null(&runner), 0);
    }
  };

  // Generate some data
  {
    const profiling::SimpleProfiler<TestLabel> profiler1(
        &(*profiler_data[0])[1], 1, 1);
    wait();
    const profiling::SimpleProfiler<TestLabel> profiler2(
        &(*profiler_data[0])[1], 1, 2);
    wait();
  }
  {
    const profiling::SimpleProfiler<TestLabel> profiler1(
        &(*profiler_data[0])[2], 2, 1);
    wait();
  }

  do_observe(-1.2);

  CHECK(file.list_dats().size() == 8);
  for (size_t proc = 0; proc < 2 * cores_per_node; ++proc) {
    check_unprofiled(file, proc, {-1.2});
  }
  check_test_label(file, 1, 1,
                   {{-1.2, {{Expected::Unequal, Expected::Equal}}}});
  check_test_label(file, 2, 2, {{-1.2, {{Expected::Equal, Expected::Zero}}}});

  // Add some more data
  {
    const profiling::SimpleProfiler<TestLabel> profiler1(
        &(*profiler_data[0])[2], 1, 2);
    wait();
  }

  do_observe(0.31);

  CHECK(file.list_dats().size() == 9);
  for (size_t proc = 0; proc < 2 * cores_per_node; ++proc) {
    check_unprofiled(file, proc, {-1.2, 0.31});
  }
  check_test_label(file, 1, 1,
                   {{-1.2, {{Expected::Unequal, Expected::Equal}}},
                    {0.31, {{Expected::Unequal, Expected::Equal}}}});
  check_test_label(file, 2, 2,
                   {{-1.2, {{Expected::Equal, Expected::Zero}}},
                    {0.31, {{Expected::Equal, Expected::Zero}}}});
  check_test_label(file, 2, 1, {{0.31, {{Expected::Zero, Expected::Equal}}}});
}
}  // namespace
