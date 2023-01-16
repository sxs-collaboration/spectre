// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Helpers/IO/VolumeData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/YlmSpherepack.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/FileSystem.hpp"

namespace {
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
      {"InertialCoordinates_x", tensor_and_coord_data[0]},
      {"InertialCoordinates_y", tensor_and_coord_data[1]},
      {"InertialCoordinates_z", tensor_and_coord_data[2]},
      {"TestScalar", tensor_and_coord_data[3]}};

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
            {grid_name, tensor_components, extents, bases, quadratures}});
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

  TestHelpers::io::VolumeData::check_volume_data(
      h5_file_name, version_number, "element_data"s, observation_ids[0],
      observation_values[0], tensor_and_coord_data, {{grid_name}}, {bases},
      {quadratures}, {extents},
      {"InertialCoordinates_x", "InertialCoordinates_y",
       "InertialCoordinates_z", "TestScalar"},
      {{0, 1, 2, 3}}, {}, observation_values[0]);

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
  DataType extra_tensor_component(16);
  std::iota(extra_tensor_component.begin(), extra_tensor_component.end(), 1.);
  const std::vector<size_t> observation_ids{8435087234, size_t(-1)};
  const std::vector<double> observation_values{8.0, -2.3};
  const std::vector<std::string> grid_names{"[[2,3,4]]", "[[5,6,7]]"};
  const std::vector<std::vector<Spectral::Basis>> bases{
      {3, Spectral::Basis::Chebyshev}, {3, Spectral::Basis::Legendre}};
  const std::vector<std::vector<Spectral::Quadrature>> quadratures{
      {3, Spectral::Quadrature::Gauss},
      {3, Spectral::Quadrature::GaussLobatto}};
  const auto domain_creator = domain::creators::Brick{
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
  {
    auto& volume_file =
        my_file.insert<h5::VolumeData>("/element_data", version_number);
    const auto write_to_file = [&volume_file, &tensor_components_and_coords,
                                &grid_names, &bases, &quadratures,
                                &domain_creator, &extra_tensor_component](
                                   const size_t observation_id,
                                   const double observation_value) {
      std::string first_grid = grid_names.front();
      std::string last_grid = grid_names.back();
      volume_file.write_volume_data(
          observation_id, observation_value,
          std::vector<ElementVolumeData>{
              {first_grid,
               {TensorComponent{"S", TestHelpers::io::VolumeData::multiply(
                                         observation_value,
                                         tensor_components_and_coords[0])},
                TensorComponent{
                    "x-coord",
                    TestHelpers::io::VolumeData::multiply(
                        observation_value, tensor_components_and_coords[1])},
                TensorComponent{
                    "y-coord",
                    TestHelpers::io::VolumeData::multiply(
                        observation_value, tensor_components_and_coords[2])},
                TensorComponent{
                    "z-coord",
                    TestHelpers::io::VolumeData::multiply(
                        observation_value, tensor_components_and_coords[3])},
                TensorComponent{"T_x", TestHelpers::io::VolumeData::multiply(
                                           observation_value,
                                           tensor_components_and_coords[4])},
                TensorComponent{"T_y", TestHelpers::io::VolumeData::multiply(
                                           observation_value,
                                           tensor_components_and_coords[5])},
                TensorComponent{"T_z", TestHelpers::io::VolumeData::multiply(
                                           observation_value,
                                           tensor_components_and_coords[6])}},
               {2, 2, 2},
               bases.front(),
               quadratures.front()},
              // Second Element Data
              {last_grid,
               {TensorComponent{"S", TestHelpers::io::VolumeData::multiply(
                                         observation_value,
                                         tensor_components_and_coords[1])},
                TensorComponent{
                    "x-coord",
                    TestHelpers::io::VolumeData::multiply(
                        observation_value, tensor_components_and_coords[0])},
                TensorComponent{
                    "y-coord",
                    TestHelpers::io::VolumeData::multiply(
                        observation_value, tensor_components_and_coords[5])},
                TensorComponent{
                    "z-coord",
                    TestHelpers::io::VolumeData::multiply(
                        observation_value, tensor_components_and_coords[3])},
                TensorComponent{"T_x", TestHelpers::io::VolumeData::multiply(
                                           observation_value,
                                           tensor_components_and_coords[6])},
                TensorComponent{"T_y", TestHelpers::io::VolumeData::multiply(
                                           observation_value,
                                           tensor_components_and_coords[4])},
                TensorComponent{"T_z", TestHelpers::io::VolumeData::multiply(
                                           observation_value,
                                           tensor_components_and_coords[2])}},
               {2, 2, 2},
               bases.back(),
               quadratures.back()}},
          serialize(domain_creator.create_domain()),
          serialize(domain_creator.functions_of_time()));
      // Write another tensor component separately
      volume_file.write_tensor_component(observation_id, "U",
                                         extra_tensor_component);
    };
    for (size_t i = 0; i < observation_ids.size(); ++i) {
      write_to_file(observation_ids[i], observation_values[i]);
    }
    my_file.close_current_object();
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
    TestHelpers::io::VolumeData::check_volume_data(
        h5_file_name, version_number, "element_data"s, observation_ids[i],
        observation_values[i], tensor_components_and_coords, grid_names, bases,
        quadratures, {{2, 2, 2}, {2, 2, 2}},
        {"S", "x-coord", "y-coord", "z-coord", "T_x", "T_y", "T_z"},
        {{0, 1, 2, 3, 4, 5, 6}, {1, 0, 5, 3, 6, 4, 2}}, {},
        observation_values[i]);
    CHECK(volume_file.get_domain(observation_ids[i]) ==
          serialize(domain_creator.create_domain()));
    CHECK(volume_file.get_functions_of_time(observation_ids[i]) ==
          serialize(domain_creator.functions_of_time()));
    CHECK(get<DataType>(
              volume_file.get_tensor_component(observation_ids[i], "U").data) ==
          extra_tensor_component);
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
            {{"grid_name",
              {TensorComponent{"grid_name/S", DataVector{1.0, 2.0}}},
              {2},
              {Spectral::Basis::Legendre},
              {Spectral::Quadrature::Gauss}}});
      }(),
      Catch::Contains("The expected format of the tensor component names is "
                      "'COMPONENT_NAME' but found a '/' in"));
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
            {{"grid_name",
              {TensorComponent{"S", DataVector{1.0, 2.0}},
               TensorComponent{"S", DataVector{1.0, 2.0}}},
              {2},
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
            {{"grid_name",
              {TensorComponent{"S", DataVector{1.0, 2.0}}},
              {2},
              {Spectral::Basis::Legendre},
              {Spectral::Quadrature::Gauss}}});
        volume_file.find_observation_id(11.0);
      }(),
      Catch::Contains("No observation with value"));
}
