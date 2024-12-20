// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/ShellType.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::CoordinateMaps {

std::ostream& operator<<(std::ostream& os, const ShellType shell_type) {
  switch (shell_type) {
    case ShellType::Cubed:
      return os << "Cubed";
    case ShellType::Spherical:
      return os << "Spherical";
    default:
      ERROR("Unknown domain::CoordinateMaps::ShellType");
  }
}

}  // namespace domain::CoordinateMaps

template <>
domain::CoordinateMaps::ShellType
Options::create_from_yaml<domain::CoordinateMaps::ShellType>::create<void>(
    const Options::Option& options) {
  const auto shell_type = options.parse_as<std::string>();
  if (shell_type == "Cubed") {
    return domain::CoordinateMaps::ShellType::Cubed;
  } else if (shell_type == "Spherical") {
    return domain::CoordinateMaps::ShellType::Spherical;
  }
  PARSE_ERROR(options.context(), "ShellType must be 'Cubed' or 'Spherical'.");
}
