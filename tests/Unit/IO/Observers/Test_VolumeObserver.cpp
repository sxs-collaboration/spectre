// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/range/combine.hpp>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "Helpers/IO/VolumeData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"  // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"               // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "IO/Observer/VolumeActions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TaggedTuple.hpp"

// NOLINTNEXTLINE(google-build-using-namespace)
namespace helpers = TestObservers_detail;

namespace {
auto make_fake_volume_data(const observers::ArrayComponentId& id) {
  const auto hashed_id =
      static_cast<double>(std::hash<observers::ArrayComponentId>{}(id));
  std::vector<TensorComponent> data(6);
  data[0] =
      TensorComponent("T_x"s, DataVector{0.5 * hashed_id, 1.0 * hashed_id,
                                         3.0 * hashed_id, -2.0 * hashed_id});
  data[1] =
      TensorComponent("T_y"s, DataVector{-0.5 * hashed_id, -1.0 * hashed_id,
                                         -3.0 * hashed_id, 2.0 * hashed_id});

  data[2] =
      TensorComponent("S_xx"s, DataVector{10.5 * hashed_id, 11.0 * hashed_id,
                                          13.0 * hashed_id, -22.0 * hashed_id});
  data[3] = TensorComponent("S_xy"s,
                            DataVector{10.5 * hashed_id, -11.0 * hashed_id,
                                       -13.0 * hashed_id, -22.0 * hashed_id});
  data[4] =
      TensorComponent("S_yx"s, DataVector{-10.5 * hashed_id, 11.0 * hashed_id,
                                          13.0 * hashed_id, 22.0 * hashed_id});
  data[5] =
      TensorComponent("S_yy"s, DataVector{-10.5 * hashed_id, -11.0 * hashed_id,
                                          -13.0 * hashed_id, 22.0 * hashed_id});

  return std::make_tuple(Mesh<2>{{{2, 2}},
                                 Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto},
                         std::move(data));
}

// Check that WriteVolumeData correctly writes a single element of volume
// data to a file.
template <typename Metavariables, typename ObsWriter, typename ElementComp>
void check_write_volume_data(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavariables>*>
        runner,
    const ElementId<2>& element_id,
    const std::vector<std::string>& expected_tensor_names) {
  const std::string h5_write_volume_file_name{
      "./Unit.IO.Observers.VolumeObserver.WriteVolumeData"};
  const std::string h5_write_volume_group_name{"/element_data"};
  const std::string h5_write_volume_element_name{"TestElement"};

  const observers::ArrayComponentId h5_write_volume_array_id(
      std::add_pointer_t<ElementComp>{nullptr},
      Parallel::ArrayIndex<ElementId<2>>{ElementId<2>{element_id}});

  // Although the WriteVolumeData action would typically be writing
  // 2D surface "volume" data from a Strahlkorper, to simplify this test, here
  // just reuse make_fake_volume_data().
  const auto [expected_mesh, fake_volume_data] =
      make_fake_volume_data(h5_write_volume_array_id);

  const std::vector<size_t> h5_write_volume_expected_extents{
      {expected_mesh.extents(0), expected_mesh.extents(1)}};
  const std::vector<Spectral::Basis> h5_write_volume_expected_bases{
      {expected_mesh.basis(0), expected_mesh.basis(1)}};
  const std::vector<Spectral::Quadrature> h5_write_volume_expected_quadratures{
      {expected_mesh.quadrature(0), expected_mesh.quadrature(1)}};

  const observers::ObservationId write_vol_observation_id{
      1., "ElementObservationType"};

  if (file_system::check_if_file_exists(h5_write_volume_file_name)) {
    file_system::rm(h5_write_volume_file_name, true);
  }

  runner->template threaded_action<ObsWriter,
                                   observers::ThreadedActions::WriteVolumeData>(
      0, h5_write_volume_file_name, h5_write_volume_group_name,
      write_vol_observation_id,
      std::vector<ElementVolumeData>{
          {h5_write_volume_element_name, fake_volume_data,
           h5_write_volume_expected_extents, h5_write_volume_expected_bases,
           h5_write_volume_expected_quadratures}});

  {
    std::vector<DataVector> h5_write_volume_expected_tensor_data{};
    for (const auto& tensor_component : fake_volume_data) {
      h5_write_volume_expected_tensor_data.push_back(
          std::get<DataVector>(tensor_component.data));
    }

    // Expected_tensor_names order is Tx, Ty, Sxx, Syy, Sxy, Syx, but
    // h5_write_volume_tensor_data is in order Tx, Ty, Sxx, Sxy, Syx, Syy.
    // Ensuring that the tensor data components are checked in the correct order
    // determines the order of components in the last argument to
    // check_volume_data.
    TestHelpers::io::VolumeData::check_volume_data(
        h5_write_volume_file_name + ".h5"s, 0, "element_data",
        write_vol_observation_id.hash(), write_vol_observation_id.value(),
        h5_write_volume_expected_tensor_data, {h5_write_volume_element_name},
        {h5_write_volume_expected_bases},
        {h5_write_volume_expected_quadratures},
        {h5_write_volume_expected_extents}, expected_tensor_names,
        {{0, 1, 2, 5, 3, 4}}, {});
  }

  if (file_system::check_if_file_exists(h5_write_volume_file_name + ".h5"s)) {
    file_system::rm(h5_write_volume_file_name + ".h5"s, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Observers.VolumeObserver", "[Unit][Observers]") {
  using registration_list = tmpl::list<
      observers::Actions::RegisterWithObservers<
          helpers::RegisterObservers<observers::TypeOfObservation::Volume>>,
      Parallel::Actions::TerminatePhase>;

  using metavariables = helpers::Metavariables<registration_list>;
  using obs_component = helpers::observer_component<metavariables>;
  using obs_writer = helpers::observer_writer_component<metavariables>;
  using element_comp =
      helpers::element_component<metavariables, registration_list>;

  const std::string output_file_prefix = "./Unit.IO.Observers.VolumeObserver";
  const domain::creators::Brick domain_creator{
      {{0., 0., 0.}},
      {{1., 2., 3.}},
      {{1, 0, 1}},
      {{3, 4, 5}},
      {{false, false, false}},
      std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3, 0>>(
          1., std::array<double, 3>{{2., 3., 4.}})};
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  tuples::TaggedTuple<observers::Tags::ReductionFileName,
                      observers::Tags::VolumeFileName, domain::Tags::Domain<3>,
                      domain::Tags::FunctionsOfTimeInitialize>
      cache_data{"", output_file_prefix, domain_creator.create_domain(),
                 domain_creator.functions_of_time()};
  ActionTesting::MockRuntimeSystem<metavariables> runner{std::move(cache_data)};
  ActionTesting::emplace_group_component<obs_component>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<obs_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_nodegroup_component<obs_writer>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0);
  }
  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  for (const auto& id : element_ids) {
    ActionTesting::emplace_component<element_comp>(&runner, id);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Register);

  // Register elements
  for (const auto& id : element_ids) {
    ActionTesting::next_action<element_comp>(make_not_null(&runner), id);
    // Invoke the simple_action RegisterContributorWithObserver that was called
    // on the observer component by the RegisterWithObservers action.
    ActionTesting::invoke_queued_simple_action<obs_component>(
        make_not_null(&runner), 0);
  }
  // Invoke the simple_action RegisterVolumeContributorWithObserverWriter.
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const std::string h5_file_name = output_file_prefix + "0.h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  // Test passing volume data...
  const observers::ObservationId observation_id{3., "ElementObservationType"};
  for (const auto& id : element_ids) {
    const observers::ArrayComponentId array_id(
        std::add_pointer_t<element_comp>{nullptr},
        Parallel::ArrayIndex<ElementId<2>>{id});

    auto [mesh, fake_volume_data] = make_fake_volume_data(array_id);
    runner
        .simple_action<obs_component, observers::Actions::ContributeVolumeData>(
            0, observation_id, std::string{"/element_data"}, array_id,
            ElementVolumeData{id, std::move(fake_volume_data), mesh});
  }
  // Invoke the simple action 'ContributeVolumeDataToWriter'
  // to move the volume data to the Writer parallel component.
  runner.invoke_queued_threaded_action<obs_writer>(0);
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 0));

  REQUIRE(file_system::check_if_file_exists(h5_file_name));
  // Check that the H5 file was written correctly.
  h5::H5File<h5::AccessType::ReadOnly> my_file(h5_file_name);
  const auto& volume_file = my_file.get<h5::VolumeData>("/element_data");

  const auto temporal_id = observation_id.hash();
  CHECK(volume_file.list_observation_ids() == std::vector<size_t>{temporal_id});

  const auto tensor_names = volume_file.list_tensor_components(temporal_id);
  const std::vector<std::string> expected_tensor_names{"T_x",  "T_y",  "S_xx",
                                                       "S_yy", "S_xy", "S_yx"};
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
  const std::vector<std::string> grid_names =
      volume_file.get_grid_names(temporal_id);
  // A pair holds an element_id, and it's "place" in the string of grid names
  std::vector<std::tuple<ElementId<2>, size_t>> pairs;
  auto start = grid_names.begin();
  auto end = grid_names.end();
  for (const auto& element_id : element_ids) {
    const auto place =
        std::find(start, end, get_output<ElementId<2>>(element_id));
    // Check that the grid was actually found
    REQUIRE(place != end);
    // Store a tuple of the element_id and its place
    pairs.emplace_back(element_id, std::distance(start, place));
  }
  // Sort element_ids by place, this is necessary because the elements are
  // written to file in an unpredictable order, so to extract this order,
  // we look at the string of grid names, as it is written in the same order as
  // the elements.
  std::sort(pairs.begin(), pairs.end(),
            [](const std::tuple<ElementId<2>, size_t>& pair_1,
               const std::tuple<ElementId<2>, size_t>& pair_2) {
              return std::get<1>(pair_1) < std::get<1>(pair_2);
            });
  std::vector<ElementId<2>> sorted_element_ids;
  sorted_element_ids.reserve(pairs.size());
  for (const auto& pair : pairs) {
    sorted_element_ids.push_back(std::get<0>(pair));
  }
  // Read the Tensor Data that was written to the file
  std::unordered_map<std::string, DataVector> read_tensor_data;
  for (const auto& tensor_name :
       volume_file.list_tensor_components(temporal_id)) {
    read_tensor_data[tensor_name] = std::get<DataVector>(
        volume_file.get_tensor_component(temporal_id, tensor_name).data);
  }
  // Read the extents that were written to file
  const std::vector<std::vector<size_t>> read_extents =
      volume_file.get_extents(temporal_id);
  const auto read_bases = volume_file.get_bases(temporal_id);
  const auto read_quadratures = volume_file.get_quadratures(temporal_id);
  // The data is stored contiguously, and each element has a subset of the
  // data.  We need to keep track of how many points have already been checked
  // so that we know where to look in the tensor component data for the current
  // grid's data.
  size_t points_processed = 0;
  for (size_t i = 0; i < sorted_element_ids.size(); i++) {
    const auto& element_id = sorted_element_ids[i];
    const std::string grid_name = MakeString{} << element_id;
    const observers::ArrayComponentId array_id(
        std::add_pointer_t<element_comp>{nullptr},
        Parallel::ArrayIndex<ElementId<2>>{ElementId<2>{element_id}});
    const auto volume_data_fakes = make_fake_volume_data(array_id);
    // Each element contains as many data points as the product of its
    // extents, compute this number
    const size_t stride =
        alg::accumulate(std::get<0>(volume_data_fakes).extents().indices(),
                        static_cast<size_t>(1), std::multiplies<>{});

    // Check that the extents and tensor data were correctly written
    CHECK(std::vector<size_t>{std::get<0>(volume_data_fakes).extents(0),
                              std::get<0>(volume_data_fakes).extents(1)} ==
          read_extents[i]);
    for (const auto& tensor_component : std::get<1>(volume_data_fakes)) {
      CHECK(std::get<DataVector>(tensor_component.data) ==
            DataVector(
                &(read_tensor_data[tensor_component.name][points_processed]),
                stride));
    }
    points_processed += stride;
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  check_write_volume_data<metavariables, obs_writer, element_comp>(
      make_not_null(&runner), element_ids[0], expected_tensor_names);
}
