// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/AccessType.hpp"

#include <ostream>

#include "ErrorHandling/Error.hpp"

namespace h5 {
std::ostream& operator<<(std::ostream& os, const AccessType t) noexcept {
  switch (t) {
    case AccessType::ReadOnly:
      return os << "ReadOnly";
    case AccessType::ReadWrite:
      return os << "ReadWrite";
    default:
      ERROR("Unknown h5::AccessType. Known values are ReadWrite and ReadOnly.");
  }
}
}  // namespace h5
