// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace evolution::dg {
std::ostream& operator<<(std::ostream& os, const TimeSteppingPolicy value) {
  switch (value) {
    case TimeSteppingPolicy::Uninitialized:
      return os << "Uninitialized";
    case TimeSteppingPolicy::EqualRate:
      return os << "EqualRate";
    case TimeSteppingPolicy::Conservative:
      return os << "Conservative";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR(
          "An unknown value of TimeSteppingPolicy was passed to the stream "
          "operator.");
      // LCOV_EXCL_STOP
  }
}
}  // namespace evolution::dg
