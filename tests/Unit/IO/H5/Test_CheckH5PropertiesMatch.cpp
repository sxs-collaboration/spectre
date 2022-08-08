// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5PropertiesMatch.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/FileSystem.hpp"

namespace {

// Helper function to set up test files
template <typename DataType>
void setup_test_files(const std::vector<std::string>& h5_file_names) {
  REQUIRE(h5_file_names.size() == 3);

  // Sample volume data
  const std::string grid_name{"AhA"};
  const std::vector<DataType> tensor_components_and_coords{{0.0, 1.0},
                                                           {-78.9, -7.6}};
  const std::vector<TensorComponent> tensor_components{
      {"InertialCoordinates_1D", tensor_components_and_coords[0]},
      {"TestScalar", tensor_components_and_coords[1]}};

  const std::vector<size_t> observation_ids{2345, 3456, 4567};
  const std::vector<double> observation_values{1.0, 2.0, 3.0};
  const std::vector<Spectral::Basis> bases{Spectral::Basis::Legendre};
  const std::vector<Spectral::Quadrature> quadratures{
      Spectral::Quadrature::Gauss};
  const std::vector<size_t> extents{2};

  const uint32_t version_number = 4;

  // Set up files with same observation ids and source archives
  for (const auto& file_name : h5_file_names) {
    // Remove any pre-existing file with the same name
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }

    // Create a new file with the given name
    h5::H5File<h5::AccessType::ReadWrite> h5_file{file_name};
    auto& volume_file =
        h5_file.insert<h5::VolumeData>("/element_data", version_number);

    // Write volume data to new file
    for (size_t j = 0; j < 3; j++) {
      volume_file.write_volume_data(
          observation_ids[j], observation_values[j],
          std::vector<ElementVolumeData>{
              {extents, tensor_components, bases, quadratures, grid_name}});
    }
    h5_file.close_current_object();
  }

  // Braces to specify scope for H5 file
  {
    // Set up last file with additional observation id and data
    h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_names[2], true};
    auto& volume_file =
        h5_file.get<h5::VolumeData>("/element_data", version_number);
    volume_file.write_volume_data(
        5678, 4.0,
        std::vector<ElementVolumeData>{
            {extents, tensor_components, bases, quadratures, grid_name}});
  }  // End of scope for H5 file
}

template <typename DataType>
void test_check_src_files_match() {
  const std::vector<std::string> h5_file_names{
      "Unit.IO.H5.CheckH5PropertiesMatch0.h5",
      "Unit.IO.H5.CheckH5PropertiesMatch1.h5",
      "Unit.IO.H5.CheckH5PropertiesMatch2.h5"};

  setup_test_files<DataType>(h5_file_names);

  // Check function for files with same source archive
  CHECK(h5::check_src_files_match(h5_file_names) == true);

  // Remove files after test
  for (const auto& file_name : h5_file_names) {
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
  }
}

template <typename DataType>
void test_check_observation_ids_match() {
  // List of filenames whose ids all match
  const std::vector<std::string> h5_file_names_match{
      "Unit.IO.H5.CheckH5PropertiesMatch0.h5",
      "Unit.IO.H5.CheckH5PropertiesMatch1.h5"};

  // List of filenames whose ids do not all match
  const std::vector<std::string> h5_file_names_diff{
      "Unit.IO.H5.CheckH5PropertiesMatch0.h5",
      "Unit.IO.H5.CheckH5PropertiesMatch1.h5",
      "Unit.IO.H5.CheckH5PropertiesMatch2.h5"};

  setup_test_files<DataType>(h5_file_names_diff);

  // Check function for files with same observation ids
  CHECK(h5::check_observation_ids_match(h5_file_names_match, "/element_data") ==
        true);
  // Check function for files with different observation ids
  CHECK(h5::check_observation_ids_match(h5_file_names_diff, "/element_data") ==
        false);

  // Remove files after test
  for (const auto& file_name : h5_file_names_diff) {
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.H5.CheckH5PropertiesMatch", "[Unit][IO][H5]") {
  test_check_src_files_match<DataVector>();
  test_check_src_files_match<std::vector<float>>();
  test_check_observation_ids_match<DataVector>();
  test_check_observation_ids_match<std::vector<float>>();
}
