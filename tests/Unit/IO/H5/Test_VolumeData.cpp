// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <hdf5.h>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/CartoonCylinder.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/IO/VolumeData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
void test_strahlkorper() {
  constexpr size_t l_max = 8;
  constexpr double sphere_radius = 4.0;
  constexpr std::array<double, 3> center{{5.0, 6.0, 7.0}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{l_max, l_max,
                                                        sphere_radius, center};
  const ylm::Spherepack& ylm = strahlkorper.ylm_spherepack();
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

  // Check exact observation value
  TestHelpers::io::VolumeData::check_volume_data(
      h5_file_name, version_number, "element_data"s, observation_ids[0],
      observation_values[0], std::nullopt, tensor_and_coord_data, {{grid_name}},
      {bases}, {quadratures}, {extents},
      {"InertialCoordinates_x", "InertialCoordinates_y",
       "InertialCoordinates_z", "TestScalar"},
      {{0, 1, 2, 3}}, {}, observation_values[0]);

  // Check observation value within epsilon
  {
    const std::optional<double> epsilon = 1.0e-8;
    TestHelpers::io::VolumeData::check_volume_data(
        h5_file_name, version_number, "element_data"s, observation_ids[0],
        observation_values[0] + 0.1 * epsilon.value(), epsilon,
        tensor_and_coord_data, {{grid_name}}, {bases}, {quadratures}, {extents},
        {"InertialCoordinates_x", "InertialCoordinates_y",
         "InertialCoordinates_z", "TestScalar"},
        {{0, 1, 2, 3}}, {}, observation_values[0]);
  }

  // Check that pole triangles are now merged into the main connectivity.
  // l_max=8 → extents=(9,17), regular quads=8*16=128, wrapping quads=8,
  // pole triangles = 2*(2*8-1) = 30.  Total cells = 166.
  // The pole triangle entries (4 ints each: tag=4, root, p2, p3) should appear
  // at the tail of the connectivity array after the quads (5 ints each).
  // clang-format off
  const std::vector<int> expected_pole_entries = {
      4, 0, 9,   18,    4, 8, 17,  26,
      4, 0, 18,  27,    4, 8, 26,  35,
      4, 0, 27,  36,    4, 8, 35,  44,
      4, 0, 36,  45,    4, 8, 44,  53,
      4, 0, 45,  54,    4, 8, 53,  62,
      4, 0, 54,  63,    4, 8, 62,  71,
      4, 0, 63,  72,    4, 8, 71,  80,
      4, 0, 72,  81,    4, 8, 80,  89,
      4, 0, 81,  90,    4, 8, 89,  98,
      4, 0, 90,  99,    4, 8, 98,  107,
      4, 0, 99,  108,   4, 8, 107, 116,
      4, 0, 108, 117,   4, 8, 116, 125,
      4, 0, 117, 126,   4, 8, 125, 134,
      4, 0, 126, 135,   4, 8, 134, 143,
      4, 0, 135, 144,   4, 8, 143, 152};
  // clang-format on

  {
    h5::H5File<h5::AccessType::ReadOnly> strahlkorper_file{h5_file_name};
    const auto& volume_file =
        strahlkorper_file.get<h5::VolumeData>("/element_data", version_number);
    const auto h5_connectivity =
        volume_file.get_tensor_component(4444, "connectivity").data;
    const auto& conn = get<0>(h5_connectivity);
    // Verify that the tail of the connectivity matches the expected pole
    // entries
    REQUIRE(conn.size() >= expected_pole_entries.size());
    const size_t tail_start = conn.size() - expected_pole_entries.size();
    for (size_t k = 0; k < expected_pole_entries.size(); ++k) {
      CHECK(static_cast<int>(conn[tail_start + k]) == expected_pole_entries[k]);
    }
    // 128 regular quads × (1+4) + 8 wrapping quads × (1+4) +
    // 30 pole triangles × (1+3) = 640 + 40 + 120 = 800
    CHECK(conn.size() == 800);
    strahlkorper_file.close_current_object();
  }

  // Verify that pole_connectivity is NOT written as a separate dataset
  {
    const h5::H5File<h5::AccessType::ReadOnly> strahlkorper_file{h5_file_name};
    const auto& volume_file =
        strahlkorper_file.get<h5::VolumeData>("/element_data", version_number);
    CHECK_THROWS_WITH(
        volume_file.get_tensor_component(4444, "pole_connectivity"),
        Catch::Matchers::ContainsSubstring("pole_connectivity"));
    strahlkorper_file.close_current_object();
  }

  // Verify element_id/block_id lengths match the total cell count, including
  // wrapping quads and pole triangles.
  // l_max=8 → extents=(9,17), regular quads=128, wrapping quads=8,
  // pole triangles=30 → 166 total cells.
  {
    const h5::H5File<h5::AccessType::ReadOnly> strahlkorper_file{h5_file_name};
    const auto& volume_file =
        strahlkorper_file.get<h5::VolumeData>("/element_data", version_number);
    const auto element_id_var =
        volume_file.get_tensor_component(4444, "ElementId").data;
    const auto& element_id = get<0>(element_id_var);
    // 128 regular + 8 wrapping + 2*(2*8-1) pole triangles = 166
    constexpr size_t expected_num_cells =
        (l_max) * (2 * l_max) + l_max + 2 * (2 * l_max - 1);
    CHECK(element_id.size() == expected_num_cells);
    const auto block_id_var =
        volume_file.get_tensor_component(4444, "BlockId").data;
    const auto& block_id = get<0>(block_id_var);
    CHECK(block_id.size() == expected_num_cells);
  }

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
  const domain::creators::Brick domain_creator{
      {{0., 0., 0.}},
      {{1., 2., 3.}},
      {{1, 0, 1}},
      {{3, 4, 5}},
      {{false, false, false}},
      {},
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
          serialize(domain_creator.functions_of_time()),
          serialize(domain_creator.functions_of_time()));
      // Write another tensor component separately
      volume_file.write_tensor_component(observation_id, "U",
                                         DataType{1., 2., 3.});
      CHECK_THROWS_WITH(volume_file.write_tensor_component(
                            observation_id, "U", extra_tensor_component),
                        Catch::Matchers::ContainsSubstring("already exists"));
      volume_file.write_tensor_component(observation_id, "U",
                                         extra_tensor_component, true);
    };
    for (size_t i = 0; i < observation_ids.size(); ++i) {
      write_to_file(observation_ids[i], observation_values[i]);
    }
    my_file.close_current_object();
  }
  // Open the read volume file and check that the observation id and values are
  // correct. No leading slash should also find the subfile, and a ".vol"
  // extension as well.
  auto& volume_file =
      my_file.get<h5::VolumeData>("element_data.vol", version_number);
  CHECK(volume_file.subfile_path() == "/element_data");
  const auto read_observation_ids = volume_file.list_observation_ids();
  // The observation IDs should be sorted by their observation value
  CHECK(read_observation_ids == std::vector<size_t>{size_t(-1), 8435087234});
  CHECK(volume_file.has_domain());
  CHECK(volume_file.has_global_functions_of_time());
  const std::string subfile_group_path =
      std::string(volume_file.subfile_path()) + h5::VolumeData::extension();
  const hid_t read_only_file_id =
      H5Fopen(h5_file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK(read_only_file_id >= 0);
  {
    const h5::detail::OpenGroup subfile_group(
        read_only_file_id, subfile_group_path, h5::AccessType::ReadOnly);
    CHECK(h5::contains_dataset_or_group(subfile_group.id(), "", "domain"));
    CHECK(h5::contains_dataset_or_group(subfile_group.id(), "",
                                        "global_functions_of_time"));
  }
  {
    const std::string observation_group_path =
        subfile_group_path + "/ObservationId" +
        std::to_string(observation_ids.front());
    const h5::detail::OpenGroup observation_group(
        read_only_file_id, observation_group_path, h5::AccessType::ReadOnly);
    CHECK_FALSE(
        h5::contains_dataset_or_group(observation_group.id(), "", "domain"));
    CHECK(h5::contains_dataset_or_group(observation_group.id(), "",
                                        "functions_of_time"));
  }
  CHECK_H5(H5Fclose(read_only_file_id), "Failed to close HDF5 file");
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
  CHECK(volume_file.get_domain() == serialize(domain_creator.create_domain()));
  for (size_t i = 0; i < observation_ids.size(); ++i) {
    TestHelpers::io::VolumeData::check_volume_data(
        h5_file_name, version_number, "element_data"s, observation_ids[i],
        observation_values[i], std::nullopt, tensor_components_and_coords,
        grid_names, bases, quadratures, {{2, 2, 2}, {2, 2, 2}},
        {"S", "x-coord", "y-coord", "z-coord", "T_x", "T_y", "T_z"},
        {{0, 1, 2, 3, 4, 5, 6}, {1, 0, 5, 3, 6, 4, 2}}, {},
        observation_values[i]);
    CHECK(volume_file.get_functions_of_time(observation_ids[i]) ==
          serialize(domain_creator.functions_of_time()));
    const auto global_fot_buffer = volume_file.get_global_functions_of_time();
    CHECK(global_fot_buffer.has_value());
    CHECK(global_fot_buffer.value() ==
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

  {
    INFO("mesh_for_grid");
    const size_t observation_id = observation_ids.front();
    const auto all_grid_names = volume_file.get_grid_names(observation_id);
    const auto all_extents = volume_file.get_extents(observation_id);
    const auto all_bases = volume_file.get_bases(observation_id);
    const auto all_quadratures = volume_file.get_quadratures(observation_id);
    const auto first_mesh =
        h5::mesh_for_grid<3>(grid_names.front(), all_grid_names, all_extents,
                             all_bases, all_quadratures);
    CHECK(first_mesh ==
          Mesh<3>(2, Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss));
    const auto last_mesh =
        h5::mesh_for_grid<3>(grid_names.back(), all_grid_names, all_extents,
                             all_bases, all_quadratures);
    CHECK(last_mesh == Mesh<3>(2, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto));
  }

  {
    INFO("Functions of time overwrite ordering");
    const size_t fot_observation_id_base = 9100;
    const double fot_observation_value_base = 12.5;
    const std::vector<ElementVolumeData> simple_element_data{
        {"[FOTGrid]",
         {TensorComponent{"SimpleScalar", DataVector(8, 1.0)}},
         {2, 2, 2},
         {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
          Spectral::Basis::Legendre},
         {Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
          Spectral::Quadrature::Gauss}}};
    const auto serialized_domain = serialize(domain_creator.create_domain());
    const auto make_serialized_functions_of_time =
        [](const double expiration_time) {
          domain::FunctionsOfTimeMap map{};
          const std::array<DataVector, 3> initial_data{
              DataVector{1.0}, DataVector{0.0}, DataVector{0.0}};
          map["Translation"] =
              std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
                  0.0, initial_data, expiration_time);
          return serialize(map);
        };
    const auto write_volume_with_data =
        [&](const size_t observation_id, const double observation_value,
            std::vector<char> serialized_functions_of_time) {
          volume_file.write_volume_data(observation_id, observation_value,
                                        simple_element_data, serialized_domain,
                                        serialized_functions_of_time,
                                        serialized_functions_of_time);
        };
    const auto read_observation_value_attribute = [&]() {
      const hid_t read_file_id =
          H5Fopen(h5_file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      CHECK(read_file_id >= 0);
      const h5::detail::OpenGroup read_subfile_group(
          read_file_id, subfile_group_path, h5::AccessType::ReadOnly);
      const auto stored_value = h5::read_value_attribute<double>(
          read_subfile_group.id(),
          "global_functions_of_time_observation_value");
      CHECK_H5(H5Fclose(read_file_id), "Failed to close HDF5 file");
      return stored_value;
    };

    const auto initial_fot = make_serialized_functions_of_time(5.0);
    write_volume_with_data(fot_observation_id_base, fot_observation_value_base,
                           initial_fot);

    auto observation_fot =
        volume_file.get_functions_of_time(fot_observation_id_base);
    CHECK(observation_fot.has_value());
    CHECK(observation_fot.value() == initial_fot);
    auto global_fot = volume_file.get_global_functions_of_time();
    CHECK(global_fot.has_value());
    CHECK(global_fot.value() == initial_fot);
    CHECK(read_observation_value_attribute() ==
          approx(fot_observation_value_base));

    const auto earlier_fot = make_serialized_functions_of_time(3.0);
    write_volume_with_data(fot_observation_id_base + 1,
                           fot_observation_value_base - 0.1, earlier_fot);
    observation_fot =
        volume_file.get_functions_of_time(fot_observation_id_base + 1);
    CHECK(observation_fot.has_value());
    // observation fot should just be written every time
    CHECK(observation_fot.value() == earlier_fot);
    global_fot = volume_file.get_global_functions_of_time();
    CHECK(global_fot.has_value());
    // global fot should not change because earlier_fot has an earlier
    // observation_value
    CHECK(global_fot.value() == initial_fot);
    CHECK(read_observation_value_attribute() ==
          approx(fot_observation_value_base));

    const auto later_fot = make_serialized_functions_of_time(7.0);
    write_volume_with_data(fot_observation_id_base + 2,
                           fot_observation_value_base + 0.1, later_fot);
    observation_fot =
        volume_file.get_functions_of_time(fot_observation_id_base + 2);
    CHECK(observation_fot.has_value());
    CHECK(observation_fot.value() == later_fot);
    global_fot = volume_file.get_global_functions_of_time();
    CHECK(global_fot.has_value());
    CHECK(global_fot.value() == later_fot);
    CHECK(read_observation_value_attribute() ==
          approx(fot_observation_value_base + 0.1));
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

template <size_t SpatialDim>
void test_extend_connectivity_data() {
  // Sample volume data
  const std::vector<size_t>& observation_ids{2345};
  const std::vector<double>& observation_values{1.0};

  // Sample data with h-ref 1 & p-ref 2 for each SpatialDim
  // Used for both number_of_elements and number_of_gridpoints
  const size_t number_of_elements = two_to_the(SpatialDim);

  // Instantiate components for write_volume_data
  std::vector<std::vector<size_t>> extents(number_of_elements,
                                           std::vector<size_t>(SpatialDim));
  std::vector<std::vector<Spectral::Basis>> bases(
      number_of_elements, std::vector<Spectral::Basis>(SpatialDim));
  std::vector<std::vector<Spectral::Quadrature>> quadratures(
      number_of_elements, std::vector<Spectral::Quadrature>(SpatialDim));
  std::vector<std::string> grid_names(number_of_elements);
  std::vector<std::vector<std::vector<float>>> tensor_components_and_coords(
      number_of_elements, std::vector<std::vector<float>>(
                              4, std::vector<float>(number_of_elements)));
  std::vector<std::vector<TensorComponent>> tensor_components(
      number_of_elements, std::vector<TensorComponent>(4));
  std::vector<ElementVolumeData> element_data(number_of_elements);

  // Base element spatial coordinates depending on SpatialDim
  switch (SpatialDim) {
    case 1:
      tensor_components_and_coords[0][0] = {0.0, 1.0};
      break;
    case 2:
      tensor_components_and_coords[0][0] = {0.0, 1.0, 0.0, 1.0};
      tensor_components_and_coords[0][1] = {0.0, 0.0, 1.0, 1.0};
      break;
    case 3:
      tensor_components_and_coords[0][0] = {0.0, 1.0, 0.0, 1.0,
                                            0.0, 1.0, 0.0, 1.0};
      tensor_components_and_coords[0][1] = {0.0, 0.0, 1.0, 1.0,
                                            0.0, 0.0, 1.0, 1.0};
      tensor_components_and_coords[0][2] = {0.0, 0.0, 0.0, 0.0,
                                            1.0, 1.0, 1.0, 1.0};
      break;
    default:
      ERROR("Invalid dimensionality");
  }

  // Populate remain element spatial coordinates
  for (size_t i = 0; i < 2; i++) {
    size_t index = i;

    for (size_t point_num = 0; point_num < number_of_elements; point_num++) {
      tensor_components_and_coords[index][0][point_num] =
          2 * i + tensor_components_and_coords[0][0][point_num];
    }
    grid_names[index] = "[B0,(L1I" + std::to_string(i) + ")]";

    if (SpatialDim > 1) {
      for (size_t j = 0; j < 2; j++) {
        index = 2 * j + i;

        for (size_t point_num = 0; point_num < number_of_elements;
             point_num++) {
          tensor_components_and_coords[index][0][point_num] =
              2 * i + tensor_components_and_coords[0][0][point_num];
          tensor_components_and_coords[index][1][point_num] =
              2 * j + tensor_components_and_coords[0][1][point_num];
        }

        grid_names[index] =
            "[B0,(L1I" + std::to_string(i) + ",L1I" + std::to_string(j) + ")]";

        if (SpatialDim == 3) {
          for (size_t k = 0; k < 2; k++) {
            index = 4 * k + 2 * j + i;

            for (size_t point_num = 0; point_num < number_of_elements;
                 point_num++) {
              tensor_components_and_coords[index][0][point_num] =
                  2 * i + tensor_components_and_coords[0][0][point_num];
              tensor_components_and_coords[index][1][point_num] =
                  2 * j + tensor_components_and_coords[0][1][point_num];
              tensor_components_and_coords[index][2][point_num] =
                  2 * k + tensor_components_and_coords[0][2][point_num];
            }
            grid_names[index] = "[B0,(L1I" + std::to_string(i) + ",L1I" +
                                std::to_string(j) + ",L1I" + std::to_string(k) +
                                ")]";
          }
        }
      }
    }
  }

  // Populate remaining components required for writing
  for (size_t i = 0; i < number_of_elements; i++) {
    for (size_t j = 0; j < SpatialDim; j++) {
      extents[i][j] = 2;
      bases[i][j] = Spectral::Basis::Legendre;
      quadratures[i][j] = Spectral::Quadrature::Gauss;
    }
    for (size_t point_num = 0; point_num < number_of_elements; point_num++) {
      tensor_components_and_coords[i][SpatialDim][point_num] =
          two_to_the(i + 1) + two_to_the(point_num);
    }
  }

  // Create TensorComponent and ElementVolumeData vector depending on SpatialDim
  for (size_t i = 0; i < number_of_elements; i++) {
    switch (SpatialDim) {
      case 1:
        tensor_components[i] = {
            {"InertialCoordinates_x", tensor_components_and_coords[i][0]},
            {"TestScalar", tensor_components_and_coords[i][1]}};
        break;
      case 2:
        tensor_components[i] = {
            {"InertialCoordinates_x", tensor_components_and_coords[i][0]},
            {"InertialCoordinates_y", tensor_components_and_coords[i][1]},
            {"TestScalar", tensor_components_and_coords[i][2]}};
        break;
      case 3:
        tensor_components[i] = {
            {"InertialCoordinates_x", tensor_components_and_coords[i][0]},
            {"InertialCoordinates_y", tensor_components_and_coords[i][1]},
            {"InertialCoordinates_z", tensor_components_and_coords[i][2]},
            {"TestScalar", tensor_components_and_coords[i][3]}};
        break;
      default:
        ERROR("Invalid dimensionality");
    }

    element_data[i] = {grid_names[i], tensor_components[i], extents[i],
                       bases[i], quadratures[i]};
  }  // End of sample volume data

  const std::string h5_file_name("Unit.IO.H5.VolumeData.ExtendConnectivity.h5");
  const uint32_t version_number = 4;

  // Remove any pre-existing file with the same name
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  // Create a new file with the given name
  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name};
  auto& volume_file =
      h5_file.insert<h5::VolumeData>("/element_data", version_number);

  volume_file.write_volume_data(observation_ids[0], observation_values[0],
                                element_data);
  volume_file.extend_connectivity_data<SpatialDim>(observation_ids);

  h5_file.close_current_object();

  // Sample test connectivity. New format prepends an XDMF type tag per cell:
  // Line=2, Quad=5, Hexahedron=9. Appended after the vertex indices for ease.
  std::vector<size_t> expected_connectivity;
  switch (SpatialDim) {
    case 1:
      // 3 cells, vertex indices + 3 type tags (tag=2 for Line)
      expected_connectivity = {0, 1, 2, 3, 1, 2, 2, 2, 2};
      break;
    case 2:
      // 9 cells, vertex indices + 9 type tags (tag=5 for Quad)
      expected_connectivity = {0, 1, 3, 2, 2, 3, 9,  8,  8,  9,  11, 10,
                               1, 4, 6, 3, 3, 6, 12, 9,  9,  12, 14, 11,
                               4, 5, 7, 6, 6, 7, 13, 12, 12, 13, 15, 14,
                               5, 5, 5, 5, 5, 5, 5,  5,  5};
      break;
    case 3:
      // 27 cells (3x3x3 unique Gauss coords), vertex indices + 27 type tags
      expected_connectivity = {
          0, 1, 3, 2, 4, 5, 7, 6, 4, 5, 7, 6, 32, 33, 35, 34, 32, 33, 35, 34,
          36, 37, 39, 38, 2, 3, 17, 16, 6, 7, 21, 20, 6, 7, 21, 20, 34, 35, 49,
          48, 34, 35, 49, 48, 38, 39, 53, 52, 16, 17, 19, 18, 20, 21, 23, 22,
          20, 21, 23, 22, 48, 49, 51, 50, 48, 49, 51, 50, 52, 53, 55, 54, 1, 8,
          10, 3, 5, 12, 14, 7, 5, 12, 14, 7, 33, 40, 42, 35, 33, 40, 42, 35, 37,
          44, 46, 39, 3, 10, 24, 17, 7, 14, 28, 21, 7, 14, 28, 21, 35, 42, 56,
          49, 35, 42, 56, 49, 39, 46, 60, 53, 17, 24, 26, 19, 21, 28, 30, 23,
          21, 28, 30, 23, 49, 56, 58, 51, 49, 56, 58, 51, 53, 60, 62, 55, 8, 9,
          11, 10, 12, 13, 15, 14, 12, 13, 15, 14, 40, 41, 43, 42, 40, 41, 43,
          42, 44, 45, 47, 46, 10, 11, 25, 24, 14, 15, 29, 28, 14, 15, 29, 28,
          42, 43, 57, 56, 42, 43, 57, 56, 46, 47, 61, 60, 24, 25, 27, 26, 28,
          29, 31, 30, 28, 29, 31, 30, 56, 57, 59, 58, 56, 57, 59, 58, 60, 61,
          63, 62,
          // 27 hexahedron type tags (3x3x3 cells from 4 unique coords/dir)
          9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
          9, 9, 9, 9};
      break;
    default:
      ERROR("Invalid dimensionality");
  }  // End of sample test connectivity

  // Reopen h5 file and extract connectivity
  const auto& volume_data = h5_file.get<h5::VolumeData>("/element_data");
  const auto h5_connectivity =
      volume_data.get_tensor_component(2345, "connectivity").data;
  const auto connectivity_data = get<0>(h5_connectivity);

  // Store file connectivity in vector like expected_connectivity
  std::vector<size_t> file_connectivity(connectivity_data.size());
  for (size_t i = 0; i < connectivity_data.size(); i++) {
    file_connectivity[i] = static_cast<size_t>(connectivity_data[i]);
  }

  h5_file.close_current_object();

  // Sort to check connectivity is the same since elementwise comparison is not
  // required or accurate
  std::sort(file_connectivity.begin(), file_connectivity.end());
  std::sort(expected_connectivity.begin(), expected_connectivity.end());

  CHECK(file_connectivity == expected_connectivity);

  // Remove all the created files
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

void test_cylinder(const bool filled) {
  // filled = true -> cylinder with ZernikeB2, close the angular edges with
  // hexahedra (tag 9), fill in the inner circle with wedges (tag 8)
  // filled = false -> hollow cylinder with Fourier, just close the angular
  // edges with hexahedra (tag 9)
  // All connectivity uses the mixed-topology format: XDMF type tag per cell.
  const std::string h5_file_name("Unit.IO.H5.VolumeData.Cylinder.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  const std::string grid_name{"Cylinder"};
  const std::vector<size_t> observation_ids{99901};
  const std::vector<double> observation_values{2.2};
  const auto bases =
      filled ? std::vector<Spectral::Basis>{Spectral::Basis::ZernikeB2,
                                            Spectral::Basis::ZernikeB2,
                                            Spectral::Basis::Legendre}
             : std::vector<Spectral::Basis>{Spectral::Basis::Legendre,
                                            Spectral::Basis::Fourier,
                                            Spectral::Basis::Legendre};
  const std::vector<Spectral::Quadrature> quadratures{
      filled ? Spectral::Quadrature::GaussRadauUpper
             : Spectral::Quadrature::GaussLobatto,
      Spectral::Quadrature::Equiangular, Spectral::Quadrature::GaussLobatto};
  const std::vector<size_t> extents{2, 7, 2};

  // Not using values, just checking connectivity
  const std::vector<DataVector> tensor_and_coord_data{
      {28, 0.0}, {28, 0.0}, {28, 0.0}, {28, 0.0}};
  const std::vector<TensorComponent> tensor_components{
      {"InertialCoordinates_x", tensor_and_coord_data[0]},
      {"InertialCoordinates_y", tensor_and_coord_data[1]},
      {"InertialCoordinates_z", tensor_and_coord_data[2]},
      {"TestScalar", tensor_and_coord_data[3]}};

  {
    h5::H5File<h5::AccessType::ReadWrite> cyl_file{h5_file_name};
    auto& volume_file =
        cyl_file.insert<h5::VolumeData>("/element_data", version_number);
    volume_file.write_volume_data(
        observation_ids[0], observation_values[0],
        std::vector<ElementVolumeData>{
            {grid_name, tensor_components, extents, bases, quadratures}});
    cyl_file.close_current_object();
  }

  // Check connectivity (Mixed format: 6 regular hexahedra + 1 phi-seam
  // hexahedron, with XDMF type tags; for filled: also 5 center-fill wedges)
  {
    const h5::H5File<h5::AccessType::ReadOnly> cyl_file{h5_file_name};
    const auto& volume_file =
        cyl_file.get<h5::VolumeData>("/element_data", version_number);
    const auto h5_connectivity =
        volume_file.get_tensor_component(99901, "connectivity").data;
    const DataVector connectivity_data = get<0>(h5_connectivity);
    cyl_file.close_current_object();

    // clang-format off
    // 6 regular hexahedra (tag=9) + 1 phi-seam hexahedron (tag=9)
    DataVector expected_connectivity{
        9.,  0.,  1.,  3.,  2., 14., 15., 17., 16.,  // regular hex ip=0->1
        9.,  2.,  3.,  5.,  4., 16., 17., 19., 18.,  // regular hex ip=1->2
        9.,  4.,  5.,  7.,  6., 18., 19., 21., 20.,  // regular hex ip=2->3
        9.,  6.,  7.,  9.,  8., 20., 21., 23., 22.,  // regular hex ip=3->4
        9.,  8.,  9., 11., 10., 22., 23., 25., 24.,  // regular hex ip=4->5
        9., 10., 11., 13., 12., 24., 25., 27., 26.,  // regular hex ip=5->6
        9., 12., 13.,  1.,  0., 26., 27., 15., 14.}; // seam hex ip=6->0
    if (filled) {
      // Append 5 center-fill wedges (tag=8), one z layer (j_z=0)
      // via recursive halving of 7-point inner ring: [0,2,4,6,8,10,12]
      // (ring_hi = [14,16,18,20,22,24,26])
      expected_connectivity = DataVector{
          9.,  0.,  1.,  3.,  2., 14., 15., 17., 16.,
          9.,  2.,  3.,  5.,  4., 16., 17., 19., 18.,
          9.,  4.,  5.,  7.,  6., 18., 19., 21., 20.,
          9.,  6.,  7.,  9.,  8., 20., 21., 23., 22.,
          9.,  8.,  9., 11., 10., 22., 23., 25., 24.,
          9., 10., 11., 13., 12., 24., 25., 27., 26.,
          9., 12., 13.,  1.,  0., 26., 27., 15., 14.,
          8.,  0.,  2.,  4., 14., 16., 18.,  // wedge i=0
          8.,  4.,  6.,  8., 18., 20., 22.,  // wedge i=2
          8.,  8., 10., 12., 22., 24., 26.,  // wedge i=4
          8.,  0.,  4.,  8., 14., 18., 22.,  // halved ring wedge
          8.,  8., 12.,  0., 22., 26., 14.}; // closing wedge (even)
    }
    // clang-format on
    CHECK(connectivity_data == expected_connectivity);
  }

  // Verify ElementId and BlockId lengths match total cell count
  {
    const h5::H5File<h5::AccessType::ReadOnly> cyl_file{h5_file_name};
    const auto& volume_file =
        cyl_file.get<h5::VolumeData>("/element_data", version_number);
    // hollow: 6 regular + 1 seam = 7 cells
    // filled: 7 hexahedra + 5 wedges = 12 cells
    const size_t expected_num_cells = filled ? 12 : 7;
    const auto element_id_var =
        volume_file.get_tensor_component(99901, "ElementId").data;
    CHECK(get<0>(element_id_var).size() == expected_num_cells);
    const auto block_id_var =
        volume_file.get_tensor_component(99901, "BlockId").data;
    CHECK(get<0>(block_id_var).size() == expected_num_cells);
    const auto expected_element_id = static_cast<double>(
        static_cast<uint64_t>(std::hash<std::string>{}(grid_name)));
    for (size_t idx = 0; idx < expected_num_cells; ++idx) {
      CHECK(get<0>(element_id_var)[idx] == expected_element_id);
      CHECK(get<0>(block_id_var)[idx] == 0.0);
    }
    cyl_file.close_current_object();
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

void test_disk() {
  // The disk has some extra connectivity: one being to close the angular
  // edges, the other to fill in the inner circle
  const std::string h5_file_name("Unit.IO.H5.VolumeData.Disk.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  const std::string grid_name{"Disk"};
  const std::vector<size_t> observation_ids{12354};
  const std::vector<double> observation_values{1.1};
  const std::vector<Spectral::Basis> bases{2, Spectral::Basis::ZernikeB2};
  const std::vector<Spectral::Quadrature> quadratures{
      {Spectral::Quadrature::GaussRadauUpper,
       Spectral::Quadrature::Equiangular}};
  const std::vector<size_t> extents{2, 7};

  // Not using values, just checking connectivity
  const std::vector<DataVector> tensor_and_coord_data{
      {14, 0.0}, {14, 0.0}, {14, 0.0}};
  const std::vector<TensorComponent> tensor_components{
      {"InertialCoordinates_x", tensor_and_coord_data[0]},
      {"InertialCoordinates_y", tensor_and_coord_data[1]},
      {"TestScalar", tensor_and_coord_data[2]}};

  {
    h5::H5File<h5::AccessType::ReadWrite> disk_file{h5_file_name};
    auto& volume_file =
        disk_file.insert<h5::VolumeData>("/element_data", version_number);
    volume_file.write_volume_data(
        observation_ids[0], observation_values[0],
        std::vector<ElementVolumeData>{
            {grid_name, tensor_components, extents, bases, quadratures}});
    disk_file.close_current_object();

    // Open the read volume file and check that the observation id and values
    // are correct.
    const auto& volume_file_read =
        disk_file.get<h5::VolumeData>("/element_data", version_number);
    const auto read_observation_ids = volume_file_read.list_observation_ids();
    CHECK(read_observation_ids == std::vector<size_t>{12354});
    CHECK(volume_file_read.get_observation_value(observation_ids[0]) ==
          observation_values[0]);
  }

  // Check connectivity (Mixed format: normal quads, wrapping quad, disk
  // triangles all in one array with XDMF type tags)
  DataVector connectivity_data{};
  {
    const h5::H5File<h5::AccessType::ReadOnly> disk_file{h5_file_name};
    const auto& volume_file =
        disk_file.get<h5::VolumeData>("/element_data", version_number);
    const auto h5_connectivity =
        volume_file.get_tensor_component(12354, "connectivity").data;
    connectivity_data = get<0>(h5_connectivity);
    disk_file.close_current_object();
  }
  // clang-format off
  // 6 normal quads + 1 wrapping quad (type=5) + 5 disk triangles (type=4)
  DataVector expected_connectivity = {
     5.,  0.,  1.,  3.,  2.,   // Quad
     5.,  2.,  3.,  5.,  4.,   // Quad
     5.,  4.,  5.,  7.,  6.,   // Quad
     5.,  6.,  7.,  9.,  8.,   // Quad
     5.,  8.,  9., 11., 10.,   // Quad
     5., 10., 11., 13., 12.,   // Quad
     5.,  0.,  1., 13., 12.,   // wrapping Quad
     4.,  0.,  2.,  4.,        // Triangle
     4.,  4.,  6.,  8.,        // Triangle
     4.,  8., 10., 12.,        // Triangle
     4.,  0.,  4.,  8.,        // Triangle
     4.,  8., 12.,  0.};       // Triangle (closing)
  // clang-format on

  CHECK(connectivity_data == expected_connectivity);

  // Verify element_id and block_id datasets.
  // n_r=2, n_phi=7: 6 normal quads + 1 wrapping quad + 5 disk triangles = 12
  // cells
  {
    const h5::H5File<h5::AccessType::ReadOnly> disk_file{h5_file_name};
    const auto& volume_file =
        disk_file.get<h5::VolumeData>("/element_data", version_number);
    constexpr size_t expected_num_cells = 12;

    const auto element_id_var =
        volume_file.get_tensor_component(12354, "ElementId").data;
    const auto& element_id = get<0>(element_id_var);
    CHECK(element_id.size() == expected_num_cells);

    const auto block_id_var =
        volume_file.get_tensor_component(12354, "BlockId").data;
    const auto& block_id = get<0>(block_id_var);
    CHECK(block_id.size() == expected_num_cells);

    // All cells belong to the single "Disk" element; block_id should be 0
    // (grid name doesn't match [B<N>,... pattern)
    const auto expected_element_id = static_cast<double>(
        static_cast<uint64_t>(std::hash<std::string>{}(grid_name)));
    for (size_t i = 0; i < expected_num_cells; ++i) {
      CHECK(element_id[i] == expected_element_id);
      CHECK(block_id[i] == 0.0);
    }
    disk_file.close_current_object();
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

void test_cartoon() {
  // For a 2D computational domain Cartoon-basis evolution, we only want to
  // write 2D data despite the simulation being 3D. This tests the appropriate
  // dimension reduction
  const std::string h5_file_name("Unit.IO.H5.VolumeData.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  const std::vector<DataVector> tensor_components_and_coords{
      {8.9, 7.6, 3.9, 2.1},     {0.0, 1.0, 0.0, 1.0},
      {0.0, 0.0, 1.0, 1.0},     {0.0, 0.0, 0.0, 0.0},
      {-78.9, -7.6, -1.9, 8.1}, {-7.9, 7.6, 1.9, -8.1},
      {17.9, 27.6, 21.9, -28.1}};
  const std::vector<size_t> observation_ids{8435087234, size_t(-1)};
  const std::vector<double> observation_values{8.0, -2.3};
  const std::vector<std::string> grid_names{"[[2,3,4]]"};
  const std::vector<std::vector<Spectral::Basis>> bases{
      {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
       Spectral::Basis::Cartoon}};
  const std::vector<std::vector<Spectral::Quadrature>> quadratures{
      {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
       Spectral::Quadrature::SphericalSymmetry}};

  const std::vector<std::vector<Spectral::Basis>> written_bases{
      {Spectral::Basis::Legendre, Spectral::Basis::Legendre}};
  const std::vector<std::vector<Spectral::Quadrature>> written_quadratures{
      {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto}};

  const TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>
      test_bc{Direction<3>::lower_xi(), 0};
  const domain::creators::CartoonCylinder domain_creator{
      {0.0, 1.0},
      {1.5, 2.0},
      {0, 1},
      {5, 4},
      {domain::CoordinateMaps::Distribution::Linear,
       domain::CoordinateMaps::Distribution::Linear},
      std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3, 0>>(
          1., std::array<double, 3>{{2., 3., 4.}}),
      {{{{test_bc.get_clone(), test_bc.get_clone()}},
        {{test_bc.get_clone(), test_bc.get_clone()}}}}};

  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  {
    auto& volume_file =
        my_file.insert<h5::VolumeData>("/element_data", version_number);
    const auto write_to_file = [&volume_file, &tensor_components_and_coords,
                                &grid_names, &bases, &quadratures,
                                &domain_creator](
                                   const size_t observation_id,
                                   const double observation_value) {
      const std::string& first_grid = grid_names.front();
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
               {2, 2, 1},
               bases.front(),
               quadratures.front()}},
          serialize(domain_creator.create_domain()),
          serialize(domain_creator.functions_of_time()));
    };
    for (size_t i = 0; i < observation_ids.size(); ++i) {
      write_to_file(observation_ids[i], observation_values[i]);
    }
    my_file.close_current_object();
  }
  // Open the read volume file and check that the observation id and values are
  // correct. No leading slash should also find the subfile, and a ".vol"
  // extension as well.
  const auto& volume_file =
      my_file.get<h5::VolumeData>("element_data.vol", version_number);
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

  CHECK(volume_file.get_domain() == serialize(domain_creator.create_domain()));
  for (size_t i = 0; i < observation_ids.size(); ++i) {
    TestHelpers::io::VolumeData::check_volume_data(
        h5_file_name, version_number, "element_data"s, observation_ids[i],
        observation_values[i], std::nullopt, tensor_components_and_coords,
        grid_names, written_bases, written_quadratures, {{2, 2}},
        {"S", "x-coord", "y-coord", "z-coord", "T_x", "T_y", "T_z"},
        {{0, 1, 2, 3, 4, 5, 6}}, {}, observation_values[i]);
    CHECK(volume_file.get_functions_of_time(observation_ids[i]) ==
          serialize(domain_creator.functions_of_time()));
  }

  {
    INFO("Cartoon dimension reduction");
    const auto dimension = volume_file.get_dimension();
    CHECK(dimension == 2);
  }

  {
    INFO("Cartoon offset_and_length_for_grid");
    const size_t observation_id = observation_ids.front();
    const auto all_grid_names = volume_file.get_grid_names(observation_id);
    const auto all_extents = volume_file.get_extents(observation_id);
    const auto first_grid_offset_and_length = h5::offset_and_length_for_grid(
        grid_names.front(), all_grid_names, all_extents);
    CHECK(first_grid_offset_and_length.first == 0);
    CHECK(first_grid_offset_and_length.second == 4);
  }

  {
    INFO("Cartoon mesh_for_grid");
    const size_t observation_id = observation_ids.front();
    const auto all_grid_names = volume_file.get_grid_names(observation_id);
    const auto all_extents = volume_file.get_extents(observation_id);
    const auto all_bases = volume_file.get_bases(observation_id);
    const auto all_quadratures = volume_file.get_quadratures(observation_id);
    const auto first_mesh =
        h5::mesh_for_grid<2>(grid_names.front(), all_grid_names, all_extents,
                             all_bases, all_quadratures);
    CHECK(first_mesh ==
          Mesh<2>({2, 2},
                  {Spectral::Basis::Legendre, Spectral::Basis::Legendre},
                  {Spectral::Quadrature::GaussLobatto,
                   Spectral::Quadrature::GaussLobatto}));
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
// Test that write_volume_data produces mixed-format connectivity with XDMF type
// tags (9=Hexahedron) prepended to each cell's vertex indices.
void test_mixed_connectivity_format() {
  const std::string h5_file_name("Unit.IO.H5.VolumeData.MixedConnectivity.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
  auto& volume_file =
      h5_file.insert<h5::VolumeData>("/element_data", version_number);

  // Write a single 2x2x2 element (1 hex cell)
  volume_file.write_volume_data(
      100, 1.0,
      {{"[B0,(L0I0,L0I0,L0I0)]",
        {TensorComponent{"InertialCoordinates_x",
                         DataVector{0., 1., 0., 1., 0., 1., 0., 1.}},
         TensorComponent{"InertialCoordinates_y",
                         DataVector{0., 0., 1., 1., 0., 0., 1., 1.}},
         TensorComponent{"InertialCoordinates_z",
                         DataVector{0., 0., 0., 0., 1., 1., 1., 1.}}},
        {2, 2, 2},
        {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
         Spectral::Basis::Legendre},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
         Spectral::Quadrature::GaussLobatto}}});
  h5_file.close_current_object();

  const auto& vol_read = h5_file.get<h5::VolumeData>("/element_data");
  const auto connectivity_variant =
      vol_read.get_tensor_component(100, "connectivity").data;
  const auto& connectivity = get<0>(connectivity_variant);

  // One hex cell: [9, 0, 1, 3, 2, 4, 5, 7, 6]
  REQUIRE(connectivity.size() == 9);
  CHECK(static_cast<int>(connectivity[0]) == 9);  // XDMF Hexahedron tag

  // Write a 3x3x3 element (8 hex cells)
  h5_file.close_current_object();
  const std::string h5_file2("Unit.IO.H5.VolumeData.MixedConn3x3x3.h5");
  if (file_system::check_if_file_exists(h5_file2)) {
    file_system::rm(h5_file2, true);
  }
  {
    h5::H5File<h5::AccessType::ReadWrite> h5_file_second(h5_file2);
    auto& volume_file_second =
        h5_file_second.insert<h5::VolumeData>("/element_data", version_number);
    DataVector x(27, 0.0);
    DataVector y(27, 0.0);
    DataVector z(27, 0.0);
    for (size_t k = 0; k < 3; ++k) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 3; ++i) {
          const size_t idx = k * 9 + j * 3 + i;
          x[idx] = static_cast<double>(i);
          y[idx] = static_cast<double>(j);
          z[idx] = static_cast<double>(k);
        }
      }
    }
    volume_file_second.write_volume_data(
        200, 2.0,
        {{"[B0,(L0I0,L0I0,L0I0)]",
          {TensorComponent{"InertialCoordinates_x", x},
           TensorComponent{"InertialCoordinates_y", y},
           TensorComponent{"InertialCoordinates_z", z}},
          {3, 3, 3},
          {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
           Spectral::Basis::Legendre},
          {Spectral::Quadrature::GaussLobatto,
           Spectral::Quadrature::GaussLobatto,
           Spectral::Quadrature::GaussLobatto}}});
    h5_file_second.close_current_object();
    const auto& volume_file_second_read =
        h5_file_second.get<h5::VolumeData>("/element_data");
    const auto connectivity_3x3x3_variant =
        volume_file_second_read.get_tensor_component(200, "connectivity").data;
    const auto& connectivity_3x3x3 = get<0>(connectivity_3x3x3_variant);
    // 8 cells * (1 tag + 8 vertices) = 72
    CHECK(connectivity_3x3x3.size() == 72);
    // Each group of 9 starts with tag 9
    for (size_t i = 0; i < 8; ++i) {
      CHECK(static_cast<int>(connectivity_3x3x3[i * 9]) == 9);
    }
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  if (file_system::check_if_file_exists(h5_file2)) {
    file_system::rm(h5_file2, true);
  }
}

// Test that element_id and block_id datasets are written correctly.
void test_element_id_and_block_id() {
  const std::string h5_file_name("Unit.IO.H5.VolumeData.ElementIdBlockId.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  const std::string name0 = "[B0,(L1I0,L1I0,L1I0)]";
  const std::string name1 = "[B3,(L1I1,L1I1,L1I1)]";

  h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
  auto& volume_file =
      h5_file.insert<h5::VolumeData>("/element_data", version_number);
  volume_file.write_volume_data(
      300, 3.0,
      {{name0,
        {TensorComponent{"S", DataVector(8, 0.0)}},
        {2, 2, 2},
        {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
         Spectral::Basis::Legendre},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
         Spectral::Quadrature::GaussLobatto}},
       {name1,
        {TensorComponent{"S", DataVector(8, 1.0)}},
        {2, 2, 2},
        {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
         Spectral::Basis::Legendre},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
         Spectral::Quadrature::GaussLobatto}}});
  h5_file.close_current_object();

  const auto& volume_file_read = h5_file.get<h5::VolumeData>("/element_data");

  // Each 2x2x2 element has 1 hex cell
  const auto element_id_var =
      volume_file_read.get_tensor_component(300, "ElementId").data;
  const auto block_id_var =
      volume_file_read.get_tensor_component(300, "BlockId").data;
  const auto& element_id = get<0>(element_id_var);
  const auto& block_id = get<0>(block_id_var);

  CHECK(element_id.size() == 2);
  CHECK(block_id.size() == 2);

  // element_id = hash of element name
  const auto expected_element_id0 = ElementId<3>{name0}.to_short_id();
  const auto expected_element_id1 = ElementId<3>{name1}.to_short_id();
  CHECK(element_id[0] == approx(expected_element_id0));
  CHECK(element_id[1] == approx(expected_element_id1));

  // block_id = 0 for B0, 3 for B3
  CHECK(static_cast<uint64_t>(block_id[0]) == 0);
  CHECK(static_cast<uint64_t>(block_id[1]) == 3);

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

// Test that non-standard grid names (e.g. "AhA") cause block_id to default
// to 0 without crashing.
void test_element_id_non_standard_names() {
  const std::string h5_file_name("Unit.IO.H5.VolumeData.NonStandardNames.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
  auto& volume_file =
      h5_file.insert<h5::VolumeData>("/element_data", version_number);
  volume_file.write_volume_data(
      400, 4.0,
      {{"AhA",
        {TensorComponent{"S", DataVector(8, 0.0)}},
        {2, 2, 2},
        {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
         Spectral::Basis::Legendre},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
         Spectral::Quadrature::GaussLobatto}}});
  h5_file.close_current_object();

  const auto& volume_file_read = h5_file.get<h5::VolumeData>("/element_data");
  const auto block_id_var =
      volume_file_read.get_tensor_component(400, "BlockId").data;
  const auto& block_id = get<0>(block_id_var);
  CHECK(block_id.size() == 1);
  CHECK(static_cast<uint64_t>(block_id[0]) == 0);

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

// Test that multiple elements with different extents produce correct cell
// counts and per-element consistent element_id/block_id values.
void test_mixed_connectivity_multi_element() {
  const std::string h5_file_name("Unit.IO.H5.VolumeData.MultiElement.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  const std::string name0 = "[B0,(L0I0,L0I0,L0I0)]";  // 2x2x2 → 1 cell
  const std::string name1 = "[B1,(L0I0,L0I0,L0I0)]";  // 3x3x3 → 8 cells

  h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
  auto& volume_file =
      h5_file.insert<h5::VolumeData>("/element_data", version_number);
  volume_file.write_volume_data(
      500, 5.0,
      {{name0,
        {TensorComponent{"S", DataVector(8, 0.0)}},
        {2, 2, 2},
        {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
         Spectral::Basis::Legendre},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
         Spectral::Quadrature::GaussLobatto}},
       {name1,
        {TensorComponent{"S", DataVector(27, 0.0)}},
        {3, 3, 3},
        {Spectral::Basis::Legendre, Spectral::Basis::Legendre,
         Spectral::Basis::Legendre},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
         Spectral::Quadrature::GaussLobatto}}});
  h5_file.close_current_object();

  const auto& volume_file_read = h5_file.get<h5::VolumeData>("/element_data");

  // 1 + 8 = 9 total cells
  const auto element_id_var =
      volume_file_read.get_tensor_component(500, "ElementId").data;
  const auto block_id_var =
      volume_file_read.get_tensor_component(500, "BlockId").data;
  const auto& element_id = get<0>(element_id_var);
  const auto& block_id = get<0>(block_id_var);
  CHECK(element_id.size() == 9);
  CHECK(block_id.size() == 9);

  // First 1 cell → element 0
  const auto expected_element_id0 = ElementId<3>{name0}.to_short_id();
  const auto expected_element_id1 = ElementId<3>{name1}.to_short_id();
  CHECK(element_id[0] == approx(expected_element_id0));
  for (size_t i = 1; i < 9; ++i) {
    CHECK(element_id[i] == approx(expected_element_id1));
  }
  CHECK(static_cast<uint64_t>(block_id[0]) == 0);
  for (size_t i = 1; i < 9; ++i) {
    CHECK(static_cast<uint64_t>(block_id[i]) == 1);
  }

  // Connectivity length: 1*9 + 8*9 = 81
  const auto connectivity_var =
      volume_file_read.get_tensor_component(500, "connectivity").data;
  const auto& connectivity = get<0>(connectivity_var);
  CHECK(connectivity.size() == 81);

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}


// Test 2D annulus: Legendre+Fourier basis, extents {3,5}.
// Standard quads: (n_r-1)*(n_phi-1) = 2*4 = 8
// Wrapping quads (Fourier seam): n_r-1 = 2
// Total: 10 cells, connectivity size = 10*5 = 50
void test_annulus() {
  const std::string h5_file_name("Unit.IO.H5.VolumeData.Annulus.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  const std::string grid_name{"Annulus"};
  const std::vector<Spectral::Basis> bases{Spectral::Basis::Legendre,
                                           Spectral::Basis::Fourier};
  const std::vector<Spectral::Quadrature> quadratures{
      Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Equiangular};
  const std::vector<size_t> extents{3, 5};  // n_r=3, n_phi=5, 15 points
  const std::vector<TensorComponent> tensor_components{
      {"InertialCoordinates_x", DataVector(15, 0.0)},
      {"InertialCoordinates_y", DataVector(15, 0.0)}};

  {
    h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name};
    auto& volume_file =
        h5_file.insert<h5::VolumeData>("/element_data", version_number);
    volume_file.write_volume_data(
        12345, 1.0,
        std::vector<ElementVolumeData>{
            {grid_name, tensor_components, extents, bases, quadratures}});
    h5_file.close_current_object();
  }

  // clang-format off
  // idx(ir, ip) = ir + 3*ip  (n_r=3, n_phi=5)
  // 8 standard quads (type=5, 5 values each), then 2 wrapping quads (type=5)
  DataVector expected_connectivity = {
    // Standard quads (outer loop: first/r cells, inner loop: second/phi cells)
    5.,  0.,  1.,  4.,  3.,   // (ir=0..1, ip=0..1)
    5.,  3.,  4.,  7.,  6.,   // (ir=0..1, ip=1..2)
    5.,  6.,  7., 10.,  9.,   // (ir=0..1, ip=2..3)
    5.,  9., 10., 13., 12.,   // (ir=0..1, ip=3..4)
    5.,  1.,  2.,  5.,  4.,   // (ir=1..2, ip=0..1)
    5.,  4.,  5.,  8.,  7.,   // (ir=1..2, ip=1..2)
    5.,  7.,  8., 11., 10.,   // (ir=1..2, ip=2..3)
    5., 10., 11., 14., 13.,   // (ir=1..2, ip=3..4)
    // Wrapping quads: connect ip=(n_phi-1) back to ip=0
    5.,  0.,  1., 13., 12.,   // j=0: ir=0,1 at ip=4 → ip=0
    5.,  1.,  2., 14., 13.};  // j=1: ir=1,2 at ip=4 → ip=0
  // clang-format on

  {
    const h5::H5File<h5::AccessType::ReadOnly> h5_file{h5_file_name};
    const auto& volume_file =
        h5_file.get<h5::VolumeData>("/element_data", version_number);
    const auto connectivity_var =
        volume_file.get_tensor_component(12345, "connectivity").data;
    const auto& connectivity = get<0>(connectivity_var);
    CHECK(connectivity == expected_connectivity);

    constexpr size_t expected_cells = 10;
    const auto element_id_var =
        volume_file.get_tensor_component(12345, "ElementId").data;
    const auto& element_id = get<0>(element_id_var);
    CHECK(element_id.size() == expected_cells);

    const auto block_id_var =
        volume_file.get_tensor_component(12345, "BlockId").data;
    const auto& block_id = get<0>(block_id_var);
    CHECK(block_id.size() == expected_cells);

    // All cells belong to the same element and block 0
    const auto expected_eid =
        static_cast<double>(std::hash<std::string>{}(grid_name));
    for (size_t i = 0; i < expected_cells; ++i) {
      CHECK(element_id[i] == approx(expected_eid));
      CHECK(block_id[i] == 0.0);
    }
    h5_file.close_current_object();
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

// Test mixed disk+annulus: one ZernikeB2 disk element and one Fourier annulus.
// This mirrors the AngularDisk domain that triggered the original bug.
// Element 0 "InnerDisk": ZernikeB2+ZernikeB2, extents {2,5} → 8 cells
// Element 1 "[B1,(L0I0)]": Legendre+Fourier, extents {3,5} → 10 cells
// Total: 18 cells
void test_annulus_disk_mixed() {
  const std::string h5_file_name(
      "Unit.IO.H5.VolumeData.AnnulusDiskMixed.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  const std::string name0 = "InnerDisk";      // block_id = 0 (no [B...])
  const std::string name1 = "[B1,(L0I0,L0I0)]";  // block_id = 1

  // Element 0: ZernikeB2 disk, 2*5=10 points
  const std::vector<Spectral::Basis> disk_bases{2, Spectral::Basis::ZernikeB2};
  const std::vector<Spectral::Quadrature> disk_quadratures{
      {Spectral::Quadrature::GaussRadauUpper,
       Spectral::Quadrature::Equiangular}};
  // Element 1: Legendre+Fourier annulus, 3*5=15 points
  const std::vector<Spectral::Basis> ann_bases{Spectral::Basis::Legendre,
                                               Spectral::Basis::Fourier};
  const std::vector<Spectral::Quadrature> ann_quadratures{
      Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Equiangular};

  h5::H5File<h5::AccessType::ReadWrite> h5_file(h5_file_name);
  auto& volume_file =
      h5_file.insert<h5::VolumeData>("/element_data", version_number);
  volume_file.write_volume_data(
      5000, 2.0,
      {{name0,
        {TensorComponent{"InertialCoordinates_x", DataVector(10, 0.0)},
         TensorComponent{"InertialCoordinates_y", DataVector(10, 0.0)}},
        {2, 5},
        disk_bases,
        disk_quadratures},
       {name1,
        {TensorComponent{"InertialCoordinates_x", DataVector(15, 0.0)},
         TensorComponent{"InertialCoordinates_y", DataVector(15, 0.0)}},
        {3, 5},
        ann_bases,
        ann_quadratures}});
  h5_file.close_current_object();

  const auto& volume_file_read = h5_file.get<h5::VolumeData>("/element_data");

  // InnerDisk {2,5}: 4 standard quads + 1 wrap quad + 3 disk triangles = 8
  // [B1,(L0I0)] {3,5}: 8 standard quads + 2 wrap quads = 10
  constexpr size_t num_cells_0 = 8;
  constexpr size_t num_cells_1 = 10;
  constexpr size_t total_cells = num_cells_0 + num_cells_1;

  const auto element_id_var =
      volume_file_read.get_tensor_component(5000, "ElementId").data;
  const auto block_id_var =
      volume_file_read.get_tensor_component(5000, "BlockId").data;
  const auto& element_id = get<0>(element_id_var);
  const auto& block_id = get<0>(block_id_var);

  CHECK(element_id.size() == total_cells);
  CHECK(block_id.size() == total_cells);

  const auto expected_eid0 =
      static_cast<double>(std::hash<std::string>{}(name0));
  const auto expected_eid1 = ElementId<2>{name1}.to_short_id();
  CAPTURE(num_cells_0);
  CAPTURE(num_cells_1);
  for (size_t i = 0; i < num_cells_0; ++i) {
    CAPTURE(i);
    CHECK(element_id[i] == approx(expected_eid0));
    CHECK(block_id[i] == 0.0);
  }
  for (size_t i = num_cells_0; i < total_cells; ++i) {
    CAPTURE(i);
    CHECK(element_id[i] == expected_eid1);
    CHECK(block_id[i] == 1.0);
  }

  // Connectivity sizes:
  // InnerDisk {2,5}: 4*5 + 1*5 + 3*4 = 20+5+12 = 37
  // Annulus {3,5}:   8*5 + 2*5       = 40+10    = 50
  // Total: 87
  const auto connectivity_var =
      volume_file_read.get_tensor_component(5000, "connectivity").data;
  const auto& connectivity = get<0>(connectivity_var);
  CHECK(connectivity.size() == 87);

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
}  // namespace

// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.IO.H5.VolumeData", "[Unit][IO][H5]") {
  test<DataVector>();
  test<std::vector<float>>();
  test_cartoon();
  test_strahlkorper();
  test_disk();
  test_cylinder(true);
  test_cylinder(false);
  test_annulus();
  test_annulus_disk_mixed();
  test_extend_connectivity_data<1>();
  test_extend_connectivity_data<2>();
  test_extend_connectivity_data<3>();
  test_mixed_connectivity_format();
  test_element_id_and_block_id();
  test_element_id_non_standard_names();
  test_mixed_connectivity_multi_element();

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
      Catch::Matchers::ContainsSubstring(
          "The expected format of the tensor component names is "
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
      Catch::Matchers::ContainsSubstring(
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
      Catch::Matchers::ContainsSubstring("No observation with value"));
}
