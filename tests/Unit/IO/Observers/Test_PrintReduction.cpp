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
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestObservers_detail;

// [[OutputRegex, Current time: 3.0]]
SPECTRE_TEST_CASE("Unit.IO.Observers.PrintReduction", "[Unit][Observers]") {
  OUTPUT_TEST();

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
  ActionTesting::MockRuntimeSystem<metavariables> runner{cache_data};
  ActionTesting::emplace_component<obs_component>(&runner, 0);
  ActionTesting::next_action<obs_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<obs_writer>(&runner, 0);
  ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0);

  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  for (const auto& id : element_ids) {
    ActionTesting::emplace_component<element_comp>(&runner, id);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::RegisterWithObservers);

  // Register elements
  for (const auto& id : element_ids) {
    ActionTesting::next_action<element_comp>(make_not_null(&runner), id);
    // Invoke the simple_action RegisterContributorWithObserver that was called
    // on the observer component by the RegisterWithObservers action.
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);
  }
  // Invoke the simple_action RegisterReductionContributorWithObserverWriter.
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);

  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  const auto make_fake_reduction_data = make_overloader(
      [](const observers::ArrayComponentId& id, const double time,
         const helpers::reduction_data_from_time& /*meta*/) noexcept {
        const auto hashed_id =
            static_cast<double>(std::hash<observers::ArrayComponentId>{}(id));
        constexpr size_t number_of_grid_points = 4;
        const double error0 = 1.0e-10 * hashed_id + time;
        const double error1 = 1.0e-12 * hashed_id + 2.0 * time;
        std::string info_to_print =
        "Current time: " + std::to_string(static_cast<double>(time));
        return helpers::reduction_data_from_time{info_to_print};
      },
      [](const observers::ArrayComponentId& id, const double time,
         const helpers::reduction_data_from_vector& /*meta*/) noexcept {
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
         const helpers::reduction_data_from_ds_and_vs& /*meta*/) noexcept {
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

  tmpl::for_each<tmpl::list<helpers::reduction_data_from_time>>([
    &element_ids, &make_fake_reduction_data, &runner, &output_file_prefix
  ](auto reduction_data_v) noexcept {
    using reduction_data = tmpl::type_from<decltype(reduction_data_v)>;

    const double time = 3.0;
    const auto legend = make_overloader(
        [](const helpers::reduction_data_from_time& /*meta*/) noexcept {
          return std::vector<std::string>{"StringToPrint"};
        },
        [](const helpers::reduction_data_from_vector& /*meta*/) noexcept {
          return std::vector<std::string>{"Time", "NumberOfPoints", "Vec0",
                                          "Vec1", "Vec2"};
        },
        [](const helpers::reduction_data_from_ds_and_vs& /*meta*/) noexcept {
          return std::vector<std::string>{"Time",  "NumberOfPoints", "Error0",
                                          "Vec10", "Vec11",          "Vec20",
                                          "Vec21", "Error1"};
        })(reduction_data{});

    // Test passing reduction data.
    for (const auto& id : element_ids) {
      const observers::ArrayComponentId array_id(
          std::add_pointer_t<element_comp>{nullptr},
          Parallel::ArrayIndex<ElementId<2>>{ElementId<2>{id}});

      auto reduction_data_fakes =
          make_fake_reduction_data(array_id, time, reduction_data{});
      runner.simple_action<obs_component,
                           observers::Actions::ContributeReductionData<
                           observers::ThreadedActions::PrintReductionData>>(
          0, observers::ObservationId{time, "ElementObservationType"},
          observers::ArrayComponentId{
              std::add_pointer_t<element_comp>{nullptr},
              Parallel::ArrayIndex<typename element_comp::array_index>(id)},
          "/element_data", legend, std::move(reduction_data_fakes));
    }
    // Invoke the threaded action 'PrintReductionData' to print reduction data
    // to disk.
    runner.invoke_queued_threaded_action<obs_writer>(0);

    runner.invoke_queued_threaded_action<obs_writer>(0);
  });
}

