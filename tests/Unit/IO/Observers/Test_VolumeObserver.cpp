// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"  // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"               // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "IO/Observer/VolumeActions.hpp"  // IWYU pragma: keep
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/IO/Observers/ObserverHelpers.hpp"

// NOLINTNEXTLINE(google-build-using-namespace)
namespace helpers = TestObservers_detail;

SPECTRE_TEST_CASE("Unit.IO.Observers.VolumeObserver", "[Unit][Observers]") {
  constexpr observers::TypeOfObservation type_of_observation =
      observers::TypeOfObservation::Volume;
  using metavariables = helpers::Metavariables<type_of_observation>;
  using obs_component = helpers::observer_component<metavariables>;
  using obs_writer = helpers::observer_writer_component<metavariables>;
  using element_comp =
      helpers::element_component<metavariables, type_of_observation>;

  tuples::TaggedTuple<observers::OptionTags::ReductionFileName,
                      observers::OptionTags::VolumeFileName>
      cache_data{};
  const auto& output_file_prefix =
      tuples::get<observers::OptionTags::VolumeFileName>(cache_data) =
          "./Unit.IO.Observers.VolumeObserver";
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
  runner.set_phase(metavariables::Phase::RegisterWithObservers);

  // Register elements
  for (const auto& id : element_ids) {
    ActionTesting::next_action<element_comp>(make_not_null(&runner), id);
    // Invoke the simple_action RegisterSenderWithSelf that was called on the
    // observer component by the RegisterWithObservers action.
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);
    // Invoke the simple_action RegisterVolumeContributorWithObserverWriter.
    ActionTesting::invoke_queued_simple_action<obs_writer>(
        make_not_null(&runner), 0);
  }
  runner.set_phase(metavariables::Phase::Testing);

  const std::string h5_file_name = output_file_prefix + "0.h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  const auto make_fake_volume_data = [](const observers::ArrayComponentId& id,
                                        const std::string& element_name) {
    const auto hashed_id =
        static_cast<double>(std::hash<observers::ArrayComponentId>{}(id));
    std::vector<TensorComponent> data(6);
    data[0] = TensorComponent(element_name + "T_x"s,
                              DataVector{0.5 * hashed_id, 1.0 * hashed_id,
                                         3.0 * hashed_id, -2.0 * hashed_id});
    data[1] = TensorComponent(element_name + "T_y"s,
                              DataVector{-0.5 * hashed_id, -1.0 * hashed_id,
                                         -3.0 * hashed_id, 2.0 * hashed_id});

    data[2] = TensorComponent(element_name + "S_xx"s,
                              DataVector{10.5 * hashed_id, 11.0 * hashed_id,
                                         13.0 * hashed_id, -22.0 * hashed_id});
    data[3] = TensorComponent(element_name + "S_xy"s,
                              DataVector{10.5 * hashed_id, -11.0 * hashed_id,
                                         -13.0 * hashed_id, -22.0 * hashed_id});
    data[4] = TensorComponent(element_name + "S_yx"s,
                              DataVector{-10.5 * hashed_id, 11.0 * hashed_id,
                                         13.0 * hashed_id, 22.0 * hashed_id});
    data[5] = TensorComponent(element_name + "S_yy"s,
                              DataVector{-10.5 * hashed_id, -11.0 * hashed_id,
                                         -13.0 * hashed_id, 22.0 * hashed_id});

    return std::make_tuple(Index<2>{2, 2}, std::move(data));
  };

  // Test passing volume data...
  for (const auto& id : element_ids) {
    const observers::ArrayComponentId array_id(
        std::add_pointer_t<element_comp>{nullptr},
        Parallel::ArrayIndex<ElementIndex<2>>{ElementIndex<2>{id}});

    auto volume_data_fakes =
        make_fake_volume_data(array_id, MakeString{} << id << '/');
    runner
        .simple_action<obs_component, observers::Actions::ContributeVolumeData>(
            0,
            observers::ObservationId(
                3., typename TestObservers_detail::RegisterThisObsType<
                        type_of_observation>::ElementObservationType{}),
            std::string{"/element_data"}, array_id,
            /* get<1> = volume tensor data */
            std::move(std::get<1>(volume_data_fakes)),
            /* get<0> = index of dimensions */
            std::get<0>(volume_data_fakes));
  }
  // Invoke the simple action 'ContributeVolumeDataToWriter' to move the volume
  // data to the Writer parallel component.
  runner.invoke_queued_simple_action<obs_writer>(0);
  // Invoke the threaded action 'WriteVolumeData' to write the data to disk.
  runner.invoke_queued_threaded_action<obs_writer>(0);

  REQUIRE(file_system::check_if_file_exists(h5_file_name));
  // Check that the H5 file was written correctly.
  {
    h5::H5File<h5::AccessType::ReadOnly> my_file(h5_file_name);
    auto& volume_file = my_file.get<h5::VolumeData>("/element_data");

    const auto temporal_id =
        observers::ObservationId(
            3., typename TestObservers_detail::RegisterThisObsType<
                    type_of_observation>::ElementObservationType{})
            .hash();
    CHECK(volume_file.list_observation_ids() ==
          std::vector<size_t>{temporal_id});
    const auto grids = volume_file.list_grids(temporal_id);
    const std::vector<std::string> expected_grids(
        boost::make_transform_iterator(element_ids.begin(),
                                       get_output<ElementId<2>>),
        boost::make_transform_iterator(element_ids.end(),
                                       get_output<ElementId<2>>));
    REQUIRE(alg::all_of(grids, [&expected_grids](const std::string& name) {
      return alg::found(expected_grids, name);
    }));
    REQUIRE(alg::all_of(expected_grids, [&grids](const std::string& name) {
      return alg::found(grids, name);
    }));

    for (const auto& element_id : element_ids) {
      const std::string grid_name = MakeString{} << element_id;
      const auto tensor_names =
          volume_file.list_tensor_components(temporal_id, grid_name);
      const std::vector<std::string> expected_tensor_names{
          "T_x", "T_y", "S_xx", "S_yy", "S_xy", "S_yx"};
      CAPTURE(element_id);
      CAPTURE(tensor_names);
      CAPTURE(expected_tensor_names);
      REQUIRE(alg::all_of(tensor_names,
                          [&expected_tensor_names](const std::string& name) {
                            return alg::found(expected_tensor_names, name);
                          }));
      REQUIRE(alg::all_of(expected_tensor_names,
                          [&tensor_names](const std::string& name) {
                            return alg::found(tensor_names, name);
                          }));
      const observers::ArrayComponentId array_id(
          std::add_pointer_t<element_comp>{nullptr},
          Parallel::ArrayIndex<ElementIndex<2>>{ElementIndex<2>{element_id}});
      const auto volume_data_fakes = make_fake_volume_data(array_id, "");
      CHECK(std::vector<size_t>{std::get<0>(volume_data_fakes)[0],
                                std::get<0>(volume_data_fakes)[1]} ==
            volume_file.get_extents(temporal_id, grid_name));
      for (const auto& tensor_component : std::get<1>(volume_data_fakes)) {
        CHECK(tensor_component.data ==
              volume_file.get_tensor_component(temporal_id, grid_name,
                                               tensor_component.name));
      }
    }
  }
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
