// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Options/ParseOptions.hpp"

#include <exception>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "Informer/InfoFromBuild.hpp"
#include "Utilities/FileSystem.hpp"

namespace Options::detail {
namespace {
void check_metadata(const YAML::Node& metadata) {
  // Validate executable name
  if (const auto& exec_name = metadata["Executable"]) {
    if (file_system::get_file_name(exec_name.as<std::string>()) !=
        executable_name()) {
      throw std::runtime_error("Input file metadata lists executable '" +
                               exec_name.as<std::string>() +
                               "', but the running executable is '" +
                               executable_name() + "'.");
    }
  }
  // Validate version
  if (const auto& version = metadata["Version"]) {
    if (version.as<std::string>() != spectre_version()) {
      throw std::runtime_error(
          "Input file metadata lists version " + version.as<std::string>() +
          ", but running version " + spectre_version() + ".");
    }
  }
}
}  // namespace

YAML::Node load_and_check_yaml(const std::string& options,
                               const bool require_metadata) {
  std::vector<YAML::Node> yaml_docs = YAML::LoadAll(options);
  if (yaml_docs.empty()) {
    return {};
  } else if (yaml_docs.size() == 1) {
    if (require_metadata) {
      throw std::runtime_error(
          "Missing metadata in input file. YAML input files begin with a "
          "metadata section terminated by '---':\n\n"
          "# Metadata here\n\n"
          "---\n\n"
          "# Options start here\n\n"
          "The metadata section may also be empty:\n\n"
          "---\n"
          "---\n\n"
          "# Options start here\n\n"
          "See option parsing documentation for details.");
    }
    return yaml_docs[0];
  } else if (yaml_docs.size() == 2) {
    check_metadata(yaml_docs[0]);
    return yaml_docs[1];
  } else {
    throw std::runtime_error("Expected either one or two YAML documents, not " +
                             std::to_string(yaml_docs.size()) + ".");
  }
}
}  // namespace Options::detail
