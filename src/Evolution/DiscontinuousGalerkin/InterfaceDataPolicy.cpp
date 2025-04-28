// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace evolution::dg {
std::ostream& operator<<(std::ostream& os, const InterfaceDataPolicy value) {
  switch (value) {
    case InterfaceDataPolicy::Uninitialized:
      return os << "Uninitialized";
    case InterfaceDataPolicy::CopyProject:
      return os << "CopyProject";
    case InterfaceDataPolicy::OrientCopyProject:
      return os << "OrientCopyProject";
    case InterfaceDataPolicy::NonconformingBothInterpolate:
      return os << "NonconformingBothInterpolate";
    case InterfaceDataPolicy::NonconformingSelfInterpolates:
      return os << "NonconformingSelfInterpolates";
    case InterfaceDataPolicy::NonconformingNeighborInterpolates:
      return os << "NonconformingNeighborInterpolates";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR(
          "An unknown value of InterfaceDataPolicy was passed to the stream "
          "operator.");
      // LCOV_EXCL_STOP
  }
}
}  // namespace evolution::dg
