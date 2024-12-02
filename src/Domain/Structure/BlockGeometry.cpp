// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/BlockGeometry.hpp"

#include <ostream>
#include <string>

#include "Utilities/ErrorHandling/Error.hpp"

namespace domain {

std::ostream& operator<<(std::ostream& os, const BlockGeometry shell_type) {
  switch (shell_type) {
    case BlockGeometry::Cube:
      return os << "Cube";
    case BlockGeometry::SphericalShell:
      return os << "SphericalShell";
    default:
      ERROR("Unknown domain::BlockGeometry");
  }
}

}  // namespace domain
