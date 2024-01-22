// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cmath>
#include <iostream>
#include <string>

#include <spectre/Exporter.hpp>

int main(int argc, char** argv) {
  // Parse CLI arguments
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  if (argc != 9) {
    std::cerr << "Usage: " << argv[0]
              << " FILES_GLOB_PATTERN SUBFILE_NAME STEP FIELD X Y Z EXPECTED"
              << std::endl;
    return 1;
  }
  const std::string files_glob_pattern = argv[1];
  const std::string subfile_name = argv[2];
  const int step = std::stoi(argv[3]);
  const std::string field = argv[4];
  const double x = std::stod(argv[5]);
  const double y = std::stod(argv[6]);
  const double z = std::stod(argv[7]);
  const double expected = std::stod(argv[8]);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  // Interpolate data
  const auto interpolated_data = spectre::Exporter::interpolate_to_points<3>(
      files_glob_pattern, subfile_name, step, {field}, {{{x}, {y}, {z}}});
  const double result = interpolated_data[0][0];
  // Check result
  if (std::abs(result - expected) < 1.e-10) {
    std::cout << "SUCCESS" << std::endl;
    return 0;
  } else {
    std::cerr << "FAILURE. Result is: " << result << std::endl;
    return 1;
  }
}
