// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace Triggers {

std::ostream& operator<<(std::ostream& os, const WhenToCheck& when_to_check) {
  switch (when_to_check) {
    case WhenToCheck::AtIterations:
      os << "AtIterations";
      break;
    case WhenToCheck::AtSlabs:
      os << "AtSlabs";
      break;
    case WhenToCheck::AtSteps:
      os << "AtSteps";
      break;
    default:
      ERROR("An unknown check was passed to the stream operator. "
            << static_cast<int>(when_to_check));
  }
  return os;
}
}  // namespace Triggers
