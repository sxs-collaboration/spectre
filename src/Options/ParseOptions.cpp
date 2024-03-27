// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Options/ParseOptions.hpp"

#include <cstddef>
#include <exception>
#include <limits>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "Informer/InfoFromBuild.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"

namespace Options {
namespace detail {
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
}  // namespace detail

namespace parse_detail {
std::unordered_set<std::string> get_given_options(
    const Options::Context& context, const YAML::Node& node,
    const std::string& help) {
  if (not(node.IsMap() or node.IsNull())) {
    PARSE_ERROR(context, "'" << node << "' does not look like options.\n"
                             << help);
  }

  std::unordered_set<std::string> given_options{};
  for (const auto& name_and_value : node) {
    given_options.insert(name_and_value.first.as<std::string>());
  }
  return given_options;
}

void check_for_unique_choice(const std::vector<size_t>& alternative_choices,
                             const Options::Context& context,
                             const std::string& parsing_help) {
  if (alg::any_of(alternative_choices, [](const size_t x) {
        return x == std::numeric_limits<size_t>::max();
      })) {
    PARSE_ERROR(context, "Cannot decide between alternative options.\n"
                             << parsing_help);
  }
}

void add_name_to_valid_option_names(
    const gsl::not_null<std::vector<std::string>*> valid_option_names,
    const std::string& label) {
  ASSERT(alg::find(*valid_option_names, label) == valid_option_names->end(),
         "Duplicate option name: " << label);
  valid_option_names->push_back(label);
}

[[noreturn]] void option_specified_twice_error(
    const Options::Context& context, const std::string& name,
    const std::string& parsing_help) {
  PARSE_ERROR(context, "Option '" << name << "' specified twice.\n"
                                  << parsing_help);
}

[[noreturn]] void unused_key_error(const Context& context,
                                   const std::string& name,
                                   const std::string& parsing_help) {
  PARSE_ERROR(context, "Option '"
                           << name
                           << "' is unused because of other provided options.\n"
                           << parsing_help);
}

[[noreturn]] void option_invalid_error(const Options::Context& context,
                                       const std::string& name,
                                       const std::string& parsing_help) {
  PARSE_ERROR(context, "Option '" << name << "' is not a valid option.\n"
                                  << parsing_help);
}

void check_for_missing_option(const std::vector<std::string>& valid_names,
                              const Options::Context& context,
                              const std::string& parsing_help) {
  if (not valid_names.empty()) {
    PARSE_ERROR(context, "You did not specify the option"
                             << (valid_names.size() == 1 ? " " : "s ")
                             << (MakeString{} << valid_names) << "\n"
                             << parsing_help);
  }
}

std::string add_group_prefix_to_name(const std::string& name) {
  return "In group " + name;
}

void print_top_level_error_message() {
  Parallel::printf_error(
      "The following options differ from their suggested values:\n");
}
}  // namespace parse_detail
}  // namespace Options
