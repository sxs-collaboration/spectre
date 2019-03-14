// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Convergence/Reason.hpp"

#include <ostream>

#include "ErrorHandling/Error.hpp"

namespace Convergence {

std::ostream& operator<<(std::ostream& os, const Reason& reason) noexcept {
  switch (reason) {
    case Reason::MaxIterations:
      return os << "MaxIterations";
    case Reason::AbsoluteResidual:
      return os << "AbsoluteResidual";
    case Reason::RelativeResidual:
      return os << "RelativeResidual";
    default:
      ERROR("Unknown convergence reason");
  }
}

}  // namespace Convergence
