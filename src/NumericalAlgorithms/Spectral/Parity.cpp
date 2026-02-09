// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Parity.hpp"

#include <array>
#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace Spectral {

std::array<Parity, 3> all_parities() {
  return std::array{Parity::Uninitialized, Parity::Even, Parity::Odd};
}

std::ostream& operator<<(std::ostream& os, const Parity& parity) {
  switch (parity) {
    case Parity::Even:
      return os << "Even";
    case Parity::Odd:
      return os << "Odd";
    case Parity::Uninitialized:
      return os << "Uninitialized";
    default:
      ERROR("Unknown value of Parity trying to be streamed");
  }
}
}  // namespace Spectral
