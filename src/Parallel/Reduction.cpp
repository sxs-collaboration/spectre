// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/Reduction.hpp"

namespace Parallel {
NoSection& no_section() {
  static NoSection local_no_section{};
  return local_no_section;
}
}  // namespace Parallel
