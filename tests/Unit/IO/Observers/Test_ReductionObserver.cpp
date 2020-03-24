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
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
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

SPECTRE_TEST_CASE("Unit.IO.Observers.ReductionObserver", "[Unit][Observers]") {
  constexpr observers::TypeOfObservation type_of_observation =
      observers::TypeOfObservation::Reduction;
  using metavariables = helpers::Metavariables<type_of_observation>;
  using obs_component = helpers::observer_component<metavariables>;
  using obs_writer = helpers::observer_writer_component<metavariables>;
  using element_comp =
      helpers::element_component<metavariables, type_of_observation>;

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
    // Invoke the simple_action RegisterSenderWithSelf that was called
    // on the observer component by the RegisterWithObservers action.
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);
    // Invoke the simple_action RegisterReductionContributorWithObserverWriter.
    ActionTesting::invoke_queued_simple_action<obs_writer>(
        make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  const auto make_fake_reduction_data = make_overloader(
      [](const observers::ArrayComponentId& id, const double time,
         const helpers::reduction_data_from_doubles& /*meta*/) noexcept {
        const auto hashed_id =
            static_cast<double>(std::hash<observers::ArrayComponentId>{}(id));
        constexpr size_t number_of_grid_points = 4;
        const double error0 = 1.0e-10 * hashed_id + time;
        const double error1 = 1.0e-12 * hashed_id + 2.0 * time;
        return helpers::reduction_data_from_doubles{time, number_of_grid_points,
                                                    error0, error1};
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

  tmpl::for_each<tmpl::list<helpers::reduction_data_from_doubles,
                            helpers::reduction_data_from_vector,
                            helpers::reduction_data_from_ds_and_vs>>([
    &element_ids, &make_fake_reduction_data, &runner, &output_file_prefix
  ](auto reduction_data_v) noexcept {
    using reduction_data = tmpl::type_from<decltype(reduction_data_v)>;

    const double time = 3.0;
    const auto legend = make_overloader(
        [](const helpers::reduction_data_from_doubles& /*meta*/) noexcept {
          return std::vector<std::string>{"Time", "NumberOfPoints", "Error0",
                                          "Error1"};
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
    const std::string h5_file_name = output_file_prefix + ".h5";
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }

    // Test passing reduction data.
    for (const auto& id : element_ids) {
      const observers::ArrayComponentId array_id(
          std::add_pointer_t<element_comp>{nullptr},
          Parallel::ArrayIndex<ElementIndex<2>>{ElementIndex<2>{id}});

      auto reduction_data_fakes =
          make_fake_reduction_data(array_id, time, reduction_data{});
      runner.simple_action<obs_component,
                           observers::Actions::ContributeReductionData>(
          0,
          observers::ObservationId{
              time, typename TestObservers_detail::RegisterThisObsType<
                        type_of_observation>::ElementObservationType{}},
          "/element_data", legend, std::move(reduction_data_fakes));
    }
    // Invoke the threaded action 'WriteReductionData' to write reduction data
    // to disk.
    runner.invoke_queued_threaded_action<obs_writer>(0);

    REQUIRE(file_system::check_if_file_exists(h5_file_name));
    // Check that the H5 file was written correctly.
    {
      const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
      const auto& dat_file = file.get<h5::Dat>("/element_data");
      const Matrix written_data = dat_file.get_data();
      const auto& written_legend = dat_file.get_legend();
      CHECK(written_legend == legend);
      const auto data = make_overloader(
          [&time](
              const helpers::reduction_data_from_doubles& /*meta*/) noexcept {
            return helpers::reduction_data_from_doubles(time, 0, 0., 0.);
          },
          [&time](
              const helpers::reduction_data_from_vector& /*meta*/) noexcept {
            return helpers::reduction_data_from_vector(
                time, 0, std::vector<double>{0., 0., 0.});
          },
          [&time](
              const helpers::reduction_data_from_ds_and_vs& /*meta*/) noexcept {
            return helpers::reduction_data_from_ds_and_vs(
                time, 0, 0., std::vector<double>{0., 0.},
                std::vector<double>{0., 0.}, 0.);
          })(reduction_data{});
      const auto expected =
          alg::accumulate(
              element_ids, data,
              [&time, &make_fake_reduction_data ](
                  reduction_data state, const ElementId<2>& id) noexcept {
                const observers::ArrayComponentId array_id(
                    std::add_pointer_t<element_comp>{nullptr},
                    Parallel::ArrayIndex<ElementIndex<2>>{ElementIndex<2>{id}});
                return state.combine(
                    make_fake_reduction_data(array_id, time, reduction_data{}));
              })
              .finalize()
              .data();
      make_overloader(
          [](const auto l_expected, const auto l_written_data,
             const helpers::reduction_data_from_doubles& /*meta*/) noexcept {
            CHECK(std::get<0>(l_expected) == l_written_data(0, 0));
            CHECK(std::get<1>(l_expected) == l_written_data(0, 1));
            CHECK(std::get<2>(l_expected) == l_written_data(0, 2));
            CHECK(std::get<3>(l_expected) == l_written_data(0, 3));
          },
          [](const auto l_expected, const auto l_written_data,
             const helpers::reduction_data_from_vector&
             /*meta*/) noexcept {
            CHECK(std::get<0>(l_expected) == l_written_data(0, 0));
            CHECK(std::get<1>(l_expected) == l_written_data(0, 1));
            for (size_t i = 0; i < std::get<2>(l_expected).size(); ++i) {
              CHECK(std::get<2>(l_expected)[i] == l_written_data(0, i + 2));
            }
          },
          [](const auto l_expected, const auto l_written_data,
             const helpers::reduction_data_from_ds_and_vs&
             /*meta*/) noexcept {
            CHECK(std::get<0>(l_expected) == l_written_data(0, 0));
            CHECK(std::get<1>(l_expected) == l_written_data(0, 1));
            CHECK(std::get<2>(l_expected) == l_written_data(0, 2));
            for (size_t i = 0; i < std::get<3>(l_expected).size(); ++i) {
              CHECK(std::get<3>(l_expected)[i] == l_written_data(0, i + 3));
            }
            for (size_t i = 0; i < std::get<4>(l_expected).size(); ++i) {
              CHECK(std::get<4>(l_expected)[i] == l_written_data(0, i + 5));
            }
            CHECK(std::get<5>(l_expected) == l_written_data(0, 7));
          })(expected, written_data, reduction_data{});
    }
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
  });
}
