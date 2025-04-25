// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

#include <ostream>

namespace Spectral {

std::ostream& operator<<(std::ostream& os, SegmentSize segment_size) {
  switch (segment_size) {
    case SegmentSize::Uninitialized:
      return os << "Uninitialized";
    case SegmentSize::Full:
      return os << "Full";
    case SegmentSize::UpperHalf:
      return os << "UpperHalf";
    case SegmentSize::LowerHalf:
      return os << "LowerHalf";
    default:
      ERROR(
          "Invalid SegmentSize. Expected one of: 'Uninitialized', 'Full', "
          "'UpperHalf', 'LowerHalf'");
  }
}
}  // namespace Spectral
