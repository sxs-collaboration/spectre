// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/FaceType.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace domain {
std::ostream& operator<<(std::ostream& os, const FaceType face_type) {
  switch (face_type) {
    case FaceType::Uninitialized:
      return os << "Uninitialized";
    case FaceType::External:
      return os << "External";
    case FaceType::Topological:
      return os << "Topological";
    case FaceType::ConformingAligned:
      return os << "ConformingAligned";
    case FaceType::ConformingUnaligned:
      return os << "ConformingUnaligned";
    case FaceType::SingleNonconforming:
      return os << "SingleNonconforming";
    case FaceType::MultipleNonconforming:
      return os << "MultipleNonconforming";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("An unknown value of FaceType was passed to the stream operator.");
      // LCOV_EXCL_STOP
  }
}
}  // namespace domain
