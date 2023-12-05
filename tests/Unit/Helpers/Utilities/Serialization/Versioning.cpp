// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Utilities/Serialization/Versioning.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "Informer/InfoFromBuild.hpp"
#include "Utilities/Base64.hpp"

namespace TestHelpers::serialization::versioning_detail {
namespace {
constexpr char delimiter = ':';

std::filesystem::path full_path(const std::string& filename) {
  return std::filesystem::path(unit_test_src_path()) / filename;
}
}  // namespace

std::vector<std::pair<std::string, std::vector<std::byte>>> read_serializations(
    const std::string& filename) {
  const auto file_path = full_path(filename);
  std::ifstream file(file_path);
  if (not file) {
    return {};
  }

  std::vector<std::pair<std::string, std::vector<std::byte>>>
      serializations_to_test{};
  std::string line{};
  while (std::getline(file, line)) {
    auto label_end = line.rfind(delimiter);
    {
      INFO("Malformed version entry on line " +
           std::to_string(serializations_to_test.size() + 1));
      REQUIRE(label_end != std::string::npos);
    }
    serializations_to_test.emplace_back(
        line.substr(0, label_end), base64_decode(line.substr(label_end + 1)));
  }
  return serializations_to_test;
}

void write_serialization(const std::string& filename, const std::string& label,
                         const std::vector<std::byte>& serialization) {
  const auto file_path = full_path(filename);
  std::ofstream file(file_path, std::ios_base::app);
  {
    INFO("Failed to open " + file_path.string());
    REQUIRE(file);
  }
  file << label << delimiter << base64_encode(serialization) << "\n";
  REQUIRE(file);
  INFO("New entry written for label " + label +
       ".  Disable generate_new_entry for future runs.");
  CHECK(false);
}
}  // namespace TestHelpers::serialization::versioning_detail
