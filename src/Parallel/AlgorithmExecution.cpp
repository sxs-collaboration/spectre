// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/AlgorithmExecution.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace Parallel {
std::ostream& operator<<(std::ostream& os, const AlgorithmExecution& exec) {
  switch (exec) {
    case AlgorithmExecution::Continue:
      return os << "Continue";
    case AlgorithmExecution::Retry:
      return os << "Retry";
    case AlgorithmExecution::Pause:
      return os << "Pause";
    case AlgorithmExecution::Halt:
      return os << "Halt";
    default:
      ERROR("Unknown AlgorithmExecution: " << static_cast<int>(exec));
  }
}
}  // namespace Parallel
