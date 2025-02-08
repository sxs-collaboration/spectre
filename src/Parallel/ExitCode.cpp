// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/ExitCode.hpp"

#include <ostream>
#include <type_traits>

#include "Utilities/ErrorHandling/Error.hpp"

namespace Parallel {
std::ostream& operator<<(std::ostream& os, const ExitCode& code) {
  os << static_cast<std::underlying_type_t<ExitCode>>(code);
  switch (code) {
    case Parallel::ExitCode::Complete:
      return os << " (Complete)";
    case Parallel::ExitCode::Abort:
      return os << " (Abort)";
    case Parallel::ExitCode::ContinueFromCheckpoint:
      return os << " (ContinueFromCheckpoint)";
    default:
      ERROR("Unknown exit code: "
            << static_cast<std::underlying_type_t<ExitCode>>(code));
  }
}
}  // namespace Parallel
