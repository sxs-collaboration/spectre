// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"  // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"   // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"               // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestObservers_detail;

namespace {
// [formatter_example]
struct FormatErrors
    : tt::ConformsTo<observers::protocols::ReductionDataFormatter> {
  using reduction_data = helpers::reduction_data_from_doubles;
  std::string operator()(const double time, const size_t num_points,
                         const double error1, const double error2) const {
    return "Errors at time " + std::to_string(time) + " over " +
           std::to_string(num_points) +
           " grid points:\n  Field1: " + std::to_string(error1) +
           "\n  Field2: " + std::to_string(error2);
  }
  // NOLINTNEXTLINE
  void pup(PUP::er& /*p*/) {}
};
// [formatter_example]
static_assert(tt::assert_conforms_to<
              FormatErrors, observers::protocols::ReductionDataFormatter>);
}  // namespace

void test_reduction_observer(const bool observe_per_core) {
  using registration_list = tmpl::list<
      observers::Actions::RegisterWithObservers<
          helpers::RegisterObservers<observers::TypeOfObservation::Reduction>>,
      Parallel::Actions::TerminatePhase>;

  using metavariables = helpers::Metavariables<registration_list>;
  using obs_component = helpers::observer_component<metavariables>;
  using obs_writer = helpers::observer_writer_component<metavariables>;
  using element_comp =
      helpers::element_component<metavariables, registration_list>;

  tuples::TaggedTuple<observers::Tags::ReductionFileName,
                      observers::Tags::VolumeFileName>
      cache_data{};
  const auto& output_file_prefix =
      tuples::get<observers::Tags::ReductionFileName>(cache_data) =
          "./Unit.IO.Observers.ReductionObserver";
  const std::vector<size_t> num_cores_per_node{1, 2};
  const size_t num_cores = alg::accumulate(num_cores_per_node, size_t{0});
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      cache_data, {}, num_cores_per_node};
  ActionTesting::emplace_group_component<obs_component>(&runner);
  for (size_t core_id = 0; core_id < num_cores; ++core_id) {
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<obs_component>(make_not_null(&runner),
                                                core_id);
    }
  }
  ActionTesting::emplace_nodegroup_component<obs_writer>(&runner);
  for (size_t node_id = 0; node_id < num_cores_per_node.size(); ++node_id) {
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<obs_writer>(make_not_null(&runner), node_id);
    }
  }
  // Use the element IDs to determine on which node and core to place the
  // element. The Block ID is the node and the first segment index is the local
  // core ID on that node.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  const auto get_node_id = [](const ElementId<2>& id) { return id.block_id(); };
  const auto get_local_core_id = [](const ElementId<2>& id) {
    return id.segment_ids()[0].index();
  };
  const auto get_global_core_id = [](const ElementId<2>& id) {
    // Only works for the specific setup of {1, 2} cores per node
    return id.block_id() + id.segment_ids()[0].index();
  };
  for (const auto& id : element_ids) {
    ActionTesting::emplace_array_component<element_comp>(
        &runner, ActionTesting::NodeId{get_node_id(id)},
        ActionTesting::LocalCoreId{get_local_core_id(id)}, id);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::RegisterWithObservers);

  // Register elements
  for (const auto& id : element_ids) {
    ActionTesting::next_action<element_comp>(make_not_null(&runner), id);
    // Invoke the simple_action RegisterContributorWithObserver that was called
    // on the observer component by the RegisterWithObservers action.
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), get_global_core_id(id));
    REQUIRE(ActionTesting::is_simple_action_queue_empty<obs_component>(
        runner, get_global_core_id(id)));
  }
  // Invoke the simple_action RegisterReductionContributorWithObserverWriter.
  for (size_t node_id = 0; node_id < num_cores_per_node.size(); ++node_id) {
    ActionTesting::invoke_queued_simple_action<obs_writer>(
        make_not_null(&runner), node_id);
    ActionTesting::invoke_queued_simple_action<obs_writer>(
        make_not_null(&runner), node_id);
    REQUIRE(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner,
                                                                    node_id));
  }
  // Invoke the simple_action RegisterReductionNodeWithWritingNode.
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);
  REQUIRE(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 0));

  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  const auto make_fake_reduction_data = make_overloader(
      [](const observers::ArrayComponentId& id, const double time,
         const helpers::reduction_data_from_doubles& /*meta*/) {
        const auto hashed_id =
            static_cast<double>(std::hash<observers::ArrayComponentId>{}(id));
        constexpr size_t number_of_grid_points = 4;
        const double error0 = 1.0e-10 * hashed_id + time;
        const double error1 = 1.0e-12 * hashed_id + 2.0 * time;
        return helpers::reduction_data_from_doubles{time, number_of_grid_points,
                                                    error0, error1};
      },
      [](const observers::ArrayComponentId& id, const double time,
         const helpers::reduction_data_from_vector& /*meta*/) {
        const auto hashed_id =
            static_cast<double>(std::hash<observers::ArrayComponentId>{}(id));
        constexpr size_t number_of_grid_points = 4;
        const std::vector<double> data{1.0e-10 * hashed_id + time,
                                       1.0e-12 * hashed_id + 2.0 * time,
                                       1.0e-11 * hashed_id + 3.0 * time};
        return helpers::reduction_data_from_vector{time, number_of_grid_points,
                                                   data};
      },
      [](const observers::ArrayComponentId& id, const double time,
         const helpers::reduction_data_from_ds_and_vs& /*meta*/) {
        const auto hashed_id =
            static_cast<double>(std::hash<observers::ArrayComponentId>{}(id));
        constexpr size_t number_of_grid_points = 4;
        const double error0 = 1.0e-10 * hashed_id + time;
        const double error1 = 1.0e-12 * hashed_id + 2.0 * time;
        const std::vector<double> vector1{1.0e-10 * hashed_id + 3.0 * time,
                                          1.0e-12 * hashed_id + 4.0 * time};
        const std::vector<double> vector2{1.0e-11 * hashed_id + 5.0 * time,
                                          1.0e-13 * hashed_id + 6.0 * time};
        return helpers::reduction_data_from_ds_and_vs{
            time, number_of_grid_points, error0, vector1, vector2, error1};
      });

  using reduction_data_list =
      tmpl::list<helpers::reduction_data_from_doubles,
                 helpers::reduction_data_from_vector,
                 helpers::reduction_data_from_ds_and_vs>;
  tmpl::for_each<reduction_data_list>([&element_ids, &make_fake_reduction_data,
                                       &runner, &output_file_prefix,
                                       &observe_per_core, &num_cores_per_node,
                                       &get_global_core_id](
                                          auto reduction_data_v) {
    using reduction_data = tmpl::type_from<decltype(reduction_data_v)>;

    const double time = 3.0;
    const auto legend = []() {
      if constexpr (std::is_same_v<reduction_data,
                                   helpers::reduction_data_from_doubles>) {
        return std::vector<std::string>{"Time", "NumberOfPoints", "Error0",
                                        "Error1"};
      } else if constexpr (std::is_same_v<
                               reduction_data,
                               helpers::reduction_data_from_vector>) {
        return std::vector<std::string>{"Time", "NumberOfPoints", "Vec0",
                                        "Vec1", "Vec2"};
      } else if constexpr (std::is_same_v<
                               reduction_data,
                               helpers::reduction_data_from_ds_and_vs>) {
        return std::vector<std::string>{"Time",  "NumberOfPoints", "Error0",
                                        "Vec10", "Vec11",          "Vec20",
                                        "Vec21", "Error1"};
      }
    }();
    const std::string h5_file_name = output_file_prefix + ".h5";
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
    if (file_system::check_if_file_exists(output_file_prefix + "0.h5")) {
      file_system::rm(output_file_prefix + "0.h5", true);
    }
    if (file_system::check_if_file_exists(output_file_prefix + "1.h5")) {
      file_system::rm(output_file_prefix + "1.h5", true);
    }

    // Test passing reduction data.
    for (const auto& id : element_ids) {
      const observers::ArrayComponentId array_id(
          std::add_pointer_t<element_comp>{nullptr},
          Parallel::ArrayIndex<ElementId<2>>{ElementId<2>{id}});

      auto reduction_data_fakes =
          make_fake_reduction_data(array_id, time, reduction_data{});
      auto formatter = []() {
        if constexpr (std::is_same_v<reduction_data,
                                     helpers::reduction_data_from_doubles>) {
          return std::make_optional(FormatErrors{});
        } else {
          return std::optional<observers::NoFormatter>{std::nullopt};
        }
      }();
      runner.simple_action<obs_component,
                           observers::Actions::ContributeReductionData>(
          get_global_core_id(id),
          observers::ObservationId{time, "ElementObservationType"},
          observers::ArrayComponentId{
              std::add_pointer_t<element_comp>{nullptr},
              Parallel::ArrayIndex<typename element_comp::array_index>(id)},
          "/element_data", legend, std::move(reduction_data_fakes),
          std::move(formatter), observe_per_core);
    }
    // Invoke the threaded action 'CollectReductionDataOnNode'
    for (size_t node_id = 0; node_id < num_cores_per_node.size(); ++node_id) {
      for (size_t local_core_id = 0;
           local_core_id < num_cores_per_node.at(node_id); ++local_core_id) {
        runner.invoke_queued_threaded_action<obs_writer>(node_id);
      }
    }
    // Invoke the threaded action 'WriteReductionData' to write reduction data
    // to disk.
    runner.invoke_queued_threaded_action<obs_writer>(0);
    runner.invoke_queued_threaded_action<obs_writer>(0);
    REQUIRE(
        ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 0));

    REQUIRE(file_system::check_if_file_exists(h5_file_name));
    REQUIRE(file_system::check_if_file_exists(output_file_prefix + "0.h5") ==
            observe_per_core);
    REQUIRE(file_system::check_if_file_exists(output_file_prefix + "1.h5") ==
            observe_per_core);
    // Check that the H5 file was written correctly.
    {
      const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
      const auto& dat_file = file.get<h5::Dat>("/element_data");
      const Matrix written_data = dat_file.get_data();
      const auto& written_legend = dat_file.get_legend();
      CHECK(written_legend == legend);
      const auto data = [&time]() {
        if constexpr (std::is_same_v<reduction_data,
                                     helpers::reduction_data_from_doubles>) {
          return helpers::reduction_data_from_doubles(time, 0, 0., 0.);
        } else if constexpr (std::is_same_v<
                                 reduction_data,
                                 helpers::reduction_data_from_vector>) {
          return helpers::reduction_data_from_vector(
              time, 0, std::vector<double>{0., 0., 0.});
        } else if constexpr (std::is_same_v<
                                 reduction_data,
                                 helpers::reduction_data_from_ds_and_vs>) {
          return helpers::reduction_data_from_ds_and_vs(
              time, 0, 0., std::vector<double>{0., 0.},
              std::vector<double>{0., 0.}, 0.);
        }
      }();
      const auto expected =
          alg::accumulate(
              element_ids, data,
              [&time, &make_fake_reduction_data](reduction_data state,
                                                 const ElementId<2>& id) {
                const observers::ArrayComponentId array_id(
                    std::add_pointer_t<element_comp>{nullptr},
                    Parallel::ArrayIndex<ElementId<2>>{ElementId<2>{id}});
                return state.combine(
                    make_fake_reduction_data(array_id, time, reduction_data{}));
              })
              .finalize()
              .data();
      if constexpr (std::is_same_v<reduction_data,
                                   helpers::reduction_data_from_doubles>) {
        CHECK(std::get<0>(expected) == approx(written_data(0, 0)));
        CHECK(std::get<1>(expected) == approx(written_data(0, 1)));
        CHECK(std::get<2>(expected) == approx(written_data(0, 2)));
        CHECK(std::get<3>(expected) == approx(written_data(0, 3)));
      } else if constexpr (std::is_same_v<
                               reduction_data,
                               helpers::reduction_data_from_vector>) {
        CHECK(std::get<0>(expected) == approx(written_data(0, 0)));
        CHECK(std::get<1>(expected) == approx(written_data(0, 1)));
        for (size_t i = 0; i < std::get<2>(expected).size(); ++i) {
          CHECK(std::get<2>(expected)[i] == approx(written_data(0, i + 2)));
        }
      } else if constexpr (std::is_same_v<
                               reduction_data,
                               helpers::reduction_data_from_ds_and_vs>) {
        CHECK(std::get<0>(expected) == approx(written_data(0, 0)));
        CHECK(std::get<1>(expected) == approx(written_data(0, 1)));
        CHECK(std::get<2>(expected) == approx(written_data(0, 2)));
        for (size_t i = 0; i < std::get<3>(expected).size(); ++i) {
          CHECK(std::get<3>(expected)[i] == approx(written_data(0, i + 3)));
        }
        for (size_t i = 0; i < std::get<4>(expected).size(); ++i) {
          CHECK(std::get<4>(expected)[i] == approx(written_data(0, i + 5)));
        }
        CHECK(std::get<5>(expected) == approx(written_data(0, 7)));
      }
    }
    // Check the per-core H5 files were written correctly. Only check the
    // num-points reduction because we don't need to check here again that
    // reductions work.
    if (observe_per_core and
        std::is_same_v<reduction_data, helpers::reduction_data_from_doubles>) {
      const auto check_per_core_reduction =
          [&output_file_prefix, &legend, &time](
              const size_t node_id, const size_t global_core_id,
              const size_t expected_num_points) {
            const auto file = h5::H5File<h5::AccessType::ReadOnly>(
                output_file_prefix + std::to_string(node_id) + ".h5");
            const auto& dat_file = file.get<h5::Dat>(
                "/Core" + std ::to_string(global_core_id) + "/element_data");
            const Matrix written_data = dat_file.get_data();
            const auto& written_legend = dat_file.get_legend();
            CHECK(written_legend == legend);
            CHECK(written_data(0, 0) == approx(time));
            CHECK(written_data(0, 1) == expected_num_points);
          };
      check_per_core_reduction(0, 0, 4);
      check_per_core_reduction(1, 1, 12);
      check_per_core_reduction(1, 2, 4);
    }
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
    if (file_system::check_if_file_exists(output_file_prefix + "0.h5")) {
      file_system::rm(output_file_prefix + "0.h5", true);
    }
    if (file_system::check_if_file_exists(output_file_prefix + "1.h5")) {
      file_system::rm(output_file_prefix + "1.h5", true);
    }
  });
}

SPECTRE_TEST_CASE("Unit.IO.Observers.ReductionObserver", "[Unit][Observers]") {
  test_reduction_observer(false);
  test_reduction_observer(true);
}
