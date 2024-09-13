// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/StrahlkorperCoordsToTextFile.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/FileSystem.hpp"

namespace {
std::array<DataVector, 3> read_text_file(const std::string& filename) {
  std::array<std::vector<double>, 3> vector_result{};
  std::ifstream file(filename);
  if (not file.is_open()) {
    ERROR("Unable to open text file " << filename);
  }
  std::string line{};
  double value = 0.0;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    for (size_t i = 0; i < 3; i++) {
      ss >> value;
      gsl::at(vector_result, i).push_back(value);
    }
  }

  const size_t num_points = vector_result[0].size();

  std::array<DataVector, 3> result{
      DataVector{num_points}, DataVector{num_points}, DataVector{num_points}};

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < num_points; j++) {
      gsl::at(result, i)[j] = gsl::at(vector_result, i)[j];
    }
  }

  return result;
}

void test(const ylm::AngularOrdering ordering) {
  const std::string filename{"StrahlkorperCoords.txt"};
  const double radius = 1.5;
  const size_t l_max = 16;
  const std::array<double, 3> center{-0.1, -0.2, -0.3};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{l_max, radius, center};

  tnsr::I<DataVector, 3, Frame::Inertial> expected_points =
      ylm::cartesian_coords(strahlkorper);
  if (ordering == ylm::AngularOrdering::Cce) {
    const auto physical_extents =
        strahlkorper.ylm_spherepack().physical_extents();
    auto transpose_expected_points =
        tnsr::I<DataVector, 3, Frame::Inertial>(get<0>(expected_points).size());
    for (size_t i = 0; i < 3; ++i) {
      transpose(make_not_null(&transpose_expected_points.get(i)),
                expected_points.get(i), physical_extents[0],
                physical_extents[1]);
    }

    expected_points = std::move(transpose_expected_points);
  }

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  {
    ylm::write_strahlkorper_coords_to_text_file(strahlkorper, filename,
                                                ordering);

    std::array<DataVector, 3> points_from_file = read_text_file(filename);

    for (size_t i = 0; i < 3; i++) {
      CHECK(expected_points.get(i) == gsl::at(points_from_file, i));
    }

    CHECK_THROWS_WITH((ylm::write_strahlkorper_coords_to_text_file(
                          strahlkorper, filename, ordering)),
                      Catch::Matchers::ContainsSubstring(
                          "The output file " + filename + " already exists"));

    ylm::write_strahlkorper_coords_to_text_file(strahlkorper, filename,
                                                ordering, true);

    for (size_t i = 0; i < 3; i++) {
      CHECK(expected_points.get(i) == gsl::at(points_from_file, i));
    }
  }

  {
    ylm::write_strahlkorper_coords_to_text_file(radius, l_max, center, filename,
                                                ordering, true);

    const std::array<DataVector, 3> points_from_file = read_text_file(filename);

    for (size_t i = 0; i < 3; i++) {
      CHECK(expected_points.get(i) == gsl::at(points_from_file, i));
    }
  }

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.StrahlkorperCoordsToTextFile",
                  "[NumericalAlgorithms][Unit]") {
  test(ylm::AngularOrdering::Strahlkorper);
  test(ylm::AngularOrdering::Cce);
}
