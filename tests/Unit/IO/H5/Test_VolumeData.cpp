// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/YlmSpherepack.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"

namespace {
template <typename T>
T multiply(const double obs_value, const T& component) {
  T result = component;
  for (auto& t : result) {
    t *= obs_value;
  }
  return result;
}

// Check that the volume data is correct
template <typename DataType>
void check_volume_data(
    const std::string& h5_file_name, const uint32_t version_number,
    const size_t observation_id, const double observation_value,
    const std::vector<DataType>& tensor_components_and_coords,
    const std::vector<std::string>& grid_names,
    const std::vector<std::vector<Spectral::Basis>>& bases,
    const std::vector<std::vector<Spectral::Quadrature>>& quadratures,
    const std::vector<std::vector<size_t>>& extents,
    const std::vector<std::string>& expected_components,
    const std::vector<std::vector<size_t>>& grid_data_orders) {
  h5::H5File<h5::AccessType::ReadOnly> file_read{h5_file_name};
  const auto& volume_file =
      file_read.get<h5::VolumeData>("/element_data", version_number);

  CHECK(extents == volume_file.get_extents(observation_id));
  CHECK(volume_file.get_observation_value(observation_id) == observation_value);
  // Check that all of the grid names were written correctly by checking their
  // equality of elements
  const std::vector<std::string> read_grid_names =
      volume_file.get_grid_names(observation_id);
  [&read_grid_names, &grid_names]() {
    auto sortable_grid_names = grid_names;
    auto sortable_read_grid_names = read_grid_names;
    std::sort(sortable_grid_names.begin(), sortable_grid_names.end(),
              std::less<>{});
    std::sort(sortable_read_grid_names.begin(), sortable_read_grid_names.end(),
              std::less<>{});
    REQUIRE(sortable_read_grid_names == sortable_grid_names);
  }();
  // Find the order the grids were written in
  std::vector<size_t> grid_positions(read_grid_names.size());
  for (size_t i = 0; i < grid_positions.size(); i++) {
    auto grid_name = grid_names[i];
    auto position =
        std::find(read_grid_names.begin(), read_grid_names.end(), grid_name);
    // We know the grid name is in the read_grid_names because of the previous
    // so we know `position` is an actual pointer to an element
    grid_positions[i] =
        static_cast<size_t>(std::distance(read_grid_names.begin(), position));
  }
  auto read_bases = volume_file.get_bases(observation_id);
  alg::sort(read_bases, std::less<>{});
  auto read_quadratures = volume_file.get_quadratures(observation_id);
  alg::sort(read_quadratures, std::less<>{});
  // We need non-const bases and quadratures in order to sort them, and we
  // need them in their string form,
  const auto& stringify = [](const auto& bases_or_quadratures) {
    std::vector<std::vector<std::string>> local_target_data{};
    local_target_data.reserve(bases_or_quadratures.size() + 1);
    for (const auto& element_data : bases_or_quadratures) {
      std::vector<std::string> target_axis_data{};
      target_axis_data.reserve(element_data.size() + 1);
      for (const auto& axis_datum : element_data) {
        target_axis_data.emplace_back(MakeString{} << axis_datum);
      }
      local_target_data.push_back(target_axis_data);
    }
    return local_target_data;
  };
  auto target_bases = stringify(bases);
  alg::sort(target_bases, std::less<>{});
  auto target_quadratures = stringify(quadratures);
  alg::sort(target_quadratures, std::less<>{});
  CHECK(target_bases == read_bases);
  CHECK(target_quadratures == read_quadratures);

  const auto read_components =
      volume_file.list_tensor_components(observation_id);
  CHECK(alg::all_of(read_components,
                    [&expected_components](const std::string& id) {
                      return alg::found(expected_components, id);
                    }));
  // Helper Function to get number of points on a particular grid
  const auto accumulate_extents = [](const std::vector<size_t>& grid_extents) {
    return alg::accumulate(grid_extents, 1, std::multiplies<>{});
  };

  const auto read_extents = volume_file.get_extents(observation_id);
  std::vector<size_t> element_num_points(
      boost::make_transform_iterator(read_extents.begin(), accumulate_extents),
      boost::make_transform_iterator(read_extents.end(), accumulate_extents));
  const auto read_points_by_element = [&element_num_points]() {
    std::vector<size_t> read_points(element_num_points.size());
    read_points[0] = 0;
    for (size_t index = 1; index < element_num_points.size(); index++) {
      read_points[index] =
          read_points[index - 1] + element_num_points[index - 1];
    }
    return read_points;
  }();
  // Given a DataType, corresponding to contiguous data read out of a
  // file, find the data which was written by the grid whose extents are
  // found at position `grid_index` in the vector of extents.
  const auto get_grid_data = [&element_num_points, &read_points_by_element](
                                 const DataVector& all_data,
                                 const size_t grid_index) {
    DataType result(element_num_points[grid_index]);
    // clang-tidy: do not use pointer arithmetic
    std::copy(&all_data[read_points_by_element[grid_index]],
              &all_data[read_points_by_element[grid_index]] +  // NOLINT
                  element_num_points[grid_index],
              result.begin());
    return result;
  };
  // The tensor components can be written in any order to the file, we loop
  // over the expected components rather than the read components because they
  // are in a particular order.
  for (size_t i = 0; i < expected_components.size(); i++) {
    const auto& component = expected_components[i];
    // for each grid
    for (size_t j = 0; j < grid_names.size(); j++) {
      CHECK(get_grid_data(
                volume_file.get_tensor_component(observation_id, component),
                grid_positions[j]) ==
            multiply(observation_value,
                     tensor_components_and_coords[grid_data_orders[j][i]]));
    }
  }
}

void test_strahlkorper() {
  constexpr size_t l_max = 12;
  constexpr size_t m_max = 12;
  constexpr double sphere_radius = 4.0;
  constexpr std::array<double, 3> center{{5.0, 6.0, 7.0}};
  const Strahlkorper<Frame::Inertial> strahlkorper{l_max, m_max, sphere_radius,
                                                   center};
  const YlmSpherepack& ylm = strahlkorper.ylm_spherepack();
  const std::array<DataVector, 2> theta_phi = ylm.theta_phi_points();
  const DataVector theta = theta_phi[0];
  const DataVector phi = theta_phi[1];
  const DataVector sin_theta = sin(theta);
  const DataVector radius = ylm.spec_to_phys(strahlkorper.coefficients());
  const std::string grid_name{"AhA"};
  const std::vector<DataVector> tensor_and_coord_data{
      radius * sin_theta * cos(phi), radius * sin_theta * sin(phi),
      radius * cos(theta), cos(2.0 * theta)};
  const std::vector<TensorComponent> tensor_components{
      {grid_name + "/InertialCoordinates_x", tensor_and_coord_data[0]},
      {grid_name + "/InertialCoordinates_y", tensor_and_coord_data[1]},
      {grid_name + "/InertialCoordinates_z", tensor_and_coord_data[2]},
      {grid_name + "/TestScalar", tensor_and_coord_data[3]}};

  const std::vector<size_t> observation_ids{4444};
  const std::vector<double> observation_values{1.0};
  const std::vector<Spectral::Basis> bases{2,
                                           Spectral::Basis::SphericalHarmonic};
  const std::vector<Spectral::Quadrature> quadratures{
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}};

  const std::string h5_file_name{"Unit.IO.H5.VolumeData.Strahlkorper.h5"};
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  const std::vector<size_t> extents{
      {ylm.physical_extents()[0], ylm.physical_extents()[1]}};

  {
    h5::H5File<h5::AccessType::ReadWrite> strahlkorper_file{h5_file_name};
    auto& volume_file = strahlkorper_file.insert<h5::VolumeData>(
        "/element_data", version_number);
    volume_file.write_volume_data(
        observation_ids[0], observation_values[0],
        std::vector<ElementVolumeData>{
            {extents, tensor_components, bases, quadratures}});
    strahlkorper_file.close_current_object();

    // Open the read volume file and check that the observation id and values
    // are correct.
    const auto& volume_file_read =
        strahlkorper_file.get<h5::VolumeData>("/element_data", version_number);
    const auto read_observation_ids = volume_file_read.list_observation_ids();
    CHECK(read_observation_ids == std::vector<size_t>{4444});
    CHECK(volume_file_read.get_observation_value(observation_ids[0]) ==
          observation_values[0]);
  }

  check_volume_data(h5_file_name, version_number, observation_ids[0],
                    observation_values[0], tensor_and_coord_data, {{grid_name}},
                    {bases}, {quadratures}, {extents},
                    {"InertialCoordinates_x", "InertialCoordinates_y",
                     "InertialCoordinates_z", "TestScalar"},
                    {{0, 1, 2, 3}});

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

template <typename DataType>
void test() {
  const std::string h5_file_name("Unit.IO.H5.VolumeData.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  const std::vector<DataType> tensor_components_and_coords{
      {8.9, 7.6, 3.9, 2.1, 18.9, 17.6, 13.9, 12.1},
      {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
      {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0},
      {-78.9, -7.6, -1.9, 8.1, 6.3, 8.7, 9.8, 0.2},
      {-7.9, 7.6, 1.9, -8.1, -6.3, 2.7, 6.8, -0.2},
      {17.9, 27.6, 21.9, -28.1, -26.3, 32.7, 26.8, -30.2}};
  const std::vector<size_t> observation_ids{8435087234, size_t(-1)};
  const std::vector<double> observation_values{8.0, 2.3};
  const std::vector<std::string> grid_names{"[[2,3,4]]", "[[5,6,7]]"};
  const std::vector<std::vector<Spectral::Basis>> bases{
      {3, Spectral::Basis::Chebyshev}, {3, Spectral::Basis::Legendre}};
  const std::vector<std::vector<Spectral::Quadrature>> quadratures{
      {3, Spectral::Quadrature::Gauss},
      {3, Spectral::Quadrature::GaussLobatto}};
  {
    auto& volume_file =
        my_file.insert<h5::VolumeData>("/element_data", version_number);
    const auto write_to_file = [&volume_file, &tensor_components_and_coords,
                                &grid_names, &bases,
                                &quadratures](const size_t observation_id,
                                              const double observation_value) {
      std::string first_grid = grid_names.front();
      std::string last_grid = grid_names.back();
      volume_file.write_volume_data(
          observation_id, observation_value,
          std::vector<ElementVolumeData>{
              {{2, 2, 2},
               {TensorComponent{first_grid + "/S",
                                multiply(observation_value,
                                         tensor_components_and_coords[0])},
                TensorComponent{first_grid + "/x-coord",
                                multiply(observation_value,
                                         tensor_components_and_coords[1])},
                TensorComponent{first_grid + "/y-coord",
                                multiply(observation_value,
                                         tensor_components_and_coords[2])},
                TensorComponent{first_grid + "/z-coord",
                                multiply(observation_value,
                                         tensor_components_and_coords[3])},
                TensorComponent{first_grid + "/T_x",
                                multiply(observation_value,
                                         tensor_components_and_coords[4])},
                TensorComponent{first_grid + "/T_y",
                                multiply(observation_value,
                                         tensor_components_and_coords[5])},
                TensorComponent{first_grid + "/T_z",
                                multiply(observation_value,
                                         tensor_components_and_coords[6])}},
               bases.front(),
               quadratures.front()},
              // Second Element Data
              {{2, 2, 2},
               {TensorComponent{last_grid + "/S",
                                multiply(observation_value,
                                         tensor_components_and_coords[1])},
                TensorComponent{last_grid + "/x-coord",
                                multiply(observation_value,
                                         tensor_components_and_coords[0])},
                TensorComponent{last_grid + "/y-coord",
                                multiply(observation_value,
                                         tensor_components_and_coords[5])},
                TensorComponent{last_grid + "/z-coord",
                                multiply(observation_value,
                                         tensor_components_and_coords[3])},
                TensorComponent{last_grid + "/T_x",
                                multiply(observation_value,
                                         tensor_components_and_coords[6])},
                TensorComponent{last_grid + "/T_y",
                                multiply(observation_value,
                                         tensor_components_and_coords[4])},
                TensorComponent{last_grid + "/T_z",
                                multiply(observation_value,
                                         tensor_components_and_coords[2])}},
               bases.back(),
               quadratures.back()}});
    };
    for (size_t i = 0; i < observation_ids.size(); ++i) {
      write_to_file(observation_ids[i], observation_values[i]);
    }
  }
  // Open the read volume file and check that the observation id and values are
  // correct.
  const auto& volume_file =
      my_file.get<h5::VolumeData>("/element_data", version_number);
  CHECK(volume_file.subfile_path() == "/element_data");
  const auto read_observation_ids = volume_file.list_observation_ids();
  // The observation IDs should be sorted by their observation value
  CHECK(read_observation_ids == std::vector<size_t>{size_t(-1), 8435087234});
  {
    INFO("Test find_observation_id");
    std::vector<size_t> found_observation_ids(observation_values.size());
    std::transform(observation_values.begin(), observation_values.end(),
                   found_observation_ids.begin(),
                   [&volume_file](const double observation_value) {
                     return volume_file.find_observation_id(observation_value);
                   });
    CHECK(found_observation_ids == observation_ids);
  }

  for (size_t i = 0; i < observation_ids.size(); ++i) {
    check_volume_data(
        h5_file_name, version_number, observation_ids[i], observation_values[i],
        tensor_components_and_coords, grid_names, bases, quadratures,
        {{2, 2, 2}, {2, 2, 2}},
        {"S", "x-coord", "y-coord", "z-coord", "T_x", "T_y", "T_z"},
        {{0, 1, 2, 3, 4, 5, 6}, {1, 0, 5, 3, 6, 4, 2}});
  }

  {
    INFO("offset_and_length_for_grid");
    const size_t observation_id = observation_ids.front();
    // [find_offset]
    const auto all_grid_names = volume_file.get_grid_names(observation_id);
    const auto all_extents = volume_file.get_extents(observation_id);
    const auto first_grid_offset_and_length = h5::offset_and_length_for_grid(
        grid_names.front(), all_grid_names, all_extents);
    // [find_offset]
    CHECK(first_grid_offset_and_length.first == 0);
    CHECK(first_grid_offset_and_length.second == 8);
    const auto last_grid_offset_and_length = h5::offset_and_length_for_grid(
        grid_names.back(), all_grid_names, all_extents);
    CHECK(last_grid_offset_and_length.first == 8);
    CHECK(last_grid_offset_and_length.second == 8);
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.H5.VolumeData", "[Unit][IO][H5]") {
  test<DataVector>();
  test<std::vector<float>>();
  test_strahlkorper();

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      []() {
        const std::string h5_file_name(
            "Unit.IO.H5.VolumeData.ComponentFormat.h5");
        const uint32_t version_number = 4;
        if (file_system::check_if_file_exists(h5_file_name)) {
          file_system::rm(h5_file_name, true);
        }
        h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
        auto& volume_file =
            my_file.insert<h5::VolumeData>("/element_data", version_number);
        volume_file.write_volume_data(
            100, 10.0,
            {{{2},
              {TensorComponent{"S", DataVector{1.0, 2.0}}},
              {Spectral::Basis::Legendre},
              {Spectral::Quadrature::Gauss}}});
      }(),
      Catch::Contains(
          "The expected format of the tensor component names is "
          "'GROUP_NAME/COMPONENT_NAME' but could not find a '/' in"));
  CHECK_THROWS_WITH(
      []() {
        const std::string h5_file_name(
            "Unit.IO.H5.VolumeData.ComponentFormat1.h5");
        const uint32_t version_number = 4;
        if (file_system::check_if_file_exists(h5_file_name)) {
          file_system::rm(h5_file_name, true);
        }
        h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
        auto& volume_file =
            my_file.insert<h5::VolumeData>("/element_data", version_number);
        volume_file.write_volume_data(
            100, 10.0,
            {{{2},
              {TensorComponent{"A/S", DataVector{1.0, 2.0}},
               TensorComponent{"S", DataVector{1.0, 2.0}}},
              {Spectral::Basis::Legendre},
              {Spectral::Quadrature::Gauss}}});
      }(),
      Catch::Contains(
          "The expected format of the tensor component names is "
          "'GROUP_NAME/COMPONENT_NAME' but could not find a '/' in"));
  CHECK_THROWS_WITH(
      []() {
        const std::string h5_file_name("Unit.IO.H5.VolumeData.WriteTwice.h5");
        const uint32_t version_number = 4;
        if (file_system::check_if_file_exists(h5_file_name)) {
          file_system::rm(h5_file_name, true);
        }
        h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
        auto& volume_file =
            my_file.insert<h5::VolumeData>("/element_data", version_number);
        volume_file.write_volume_data(
            100, 10.0,
            {{{2},
              {TensorComponent{"A/S", DataVector{1.0, 2.0}},
               TensorComponent{"A/S", DataVector{1.0, 2.0}}},
              {Spectral::Basis::Legendre},
              {Spectral::Quadrature::Gauss}}});
      }(),
      Catch::Contains(
          "Trying to write tensor component 'S' which already exists in HDF5 "
          "file in group 'element_data.vol/ObservationId100'"));
#endif

  CHECK_THROWS_WITH(
      []() {
        const std::string h5_file_name(
            "Unit.IO.H5.VolumeData.FindNoObservationId.h5");
        const uint32_t version_number = 4;
        if (file_system::check_if_file_exists(h5_file_name)) {
          file_system::rm(h5_file_name, true);
        }
        h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
        auto& volume_file =
            h5_file.insert<h5::VolumeData>("/element_data", version_number);
        volume_file.write_volume_data(
            100, 10.0,
            {{{2},
              {TensorComponent{"A/S", DataVector{1.0, 2.0}}},
              {Spectral::Basis::Legendre},
              {Spectral::Quadrature::Gauss}}});
        volume_file.find_observation_id(11.0);
      }(),
      Catch::Contains("No observation with value"));
}
