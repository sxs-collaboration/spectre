// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace elliptic {
std::ostream& operator<<(
    std::ostream& os,
    const elliptic::BoundaryConditionType boundary_condition_type) noexcept {
  switch (boundary_condition_type) {
    case elliptic::BoundaryConditionType::Dirichlet:
      return os << "Dirichlet";
    case elliptic::BoundaryConditionType::Neumann:
      return os << "Neumann";
    default:
      ERROR("Missing case for operator<<(elliptic::BoundaryConditionType).");
  }
}
}  // namespace elliptic

template <>
elliptic::BoundaryConditionType
Options::create_from_yaml<elliptic::BoundaryConditionType>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Dirichlet" == type_read) {
    return elliptic::BoundaryConditionType::Dirichlet;
  } else if ("Neumann" == type_read) {
    return elliptic::BoundaryConditionType::Neumann;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to elliptic::BoundaryConditionType. Must be "
                     "either 'Dirichlet' or 'Neumann'.");
}
