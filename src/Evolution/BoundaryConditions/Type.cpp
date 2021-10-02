// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/BoundaryConditions/Type.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace evolution::BoundaryConditions {
std::ostream& operator<<(std::ostream& os, const Type boundary_condition_type) {
  switch (boundary_condition_type) {
    case Type::Ghost:
      return os << "Ghost";
    case Type::TimeDerivative:
      return os << "TimeDerivative";
    case Type::GhostAndTimeDerivative:
      return os << "GhostAndTimeDerivative";
    case Type::Outflow:
      return os << "Outflow";
    default:
      ERROR(
          "Unknown enumeration value. Should be one of Ghost, TimeDerivative, "
          "GhostAndTimeDerivative, or Outflow.");
  }
}
}  // namespace evolution::BoundaryConditions
