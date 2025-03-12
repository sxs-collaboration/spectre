// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/Topology.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace domain {
std::ostream& operator<<(std::ostream& os, const Topology topology) {
  switch (topology) {
    case Topology::Uninitialized:
      return os << "Uninitialized";
    case Topology::I1:
      return os << "I1";
    case Topology::S1:
      return os << "S1";
    case Topology::S2Colatitude:
      return os << "S2Colatitude";
    case Topology::S2Longitude:
      return os << "S2Longitude";
    case Topology::B2Radial:
      return os << "B2Radial";
    case Topology::B2Angular:
      return os << "B2Angular";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("An unknown value of Topology was passed to the stream operator.");
      // LCOV_EXCL_STOP
  }
}
}  // namespace domain
