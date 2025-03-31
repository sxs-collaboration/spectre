// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/Side.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

Side opposite(const Side side) {
  switch (side) {
    case (Side::Uninitialized):
      ERROR("Cannot get the opposite of Side::Uninitialized");
    case (Side::Lower):
      return Side::Upper;
    case (Side::Upper):
      return Side::Lower;
    case (Side::Self):
      return Side::Self;
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("An unknown Side was passed to Side::opposite()");
      // LCOV_EXCL_STOP
  }
}

std::ostream& operator<<(std::ostream& os, const Side& side) {
  switch (side) {
    case Side::Uninitialized:
      return os << "Uninitialized";
    case Side::Lower:
      return os << "Lower";
    case Side::Upper:
      return os << "Upper";
    case Side::Self:
      return os << "Self";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR(
          "A Side that was neither Upper nor Lower was passed to the stream "
          "operator.");
      // LCOV_EXCL_STOP
  }
}
