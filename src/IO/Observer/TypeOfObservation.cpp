// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/TypeOfObservation.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace observers {
std::ostream& operator<<(std::ostream& os, const TypeOfObservation& t) {
  switch (t) {
    case TypeOfObservation::Reduction:
      return os << "Reduction";
    case TypeOfObservation::Volume:
      return os << "Volume";
    default:
      ERROR("Unknown TypeOfObservation.");
  }
  return os;
}
}  // namespace observers
