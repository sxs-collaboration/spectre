// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/SimpleProfilerDataComponent.hpp"

#include <atomic>
#include <limits>

#include "Utilities/Gsl.hpp"

namespace profiling {
void InitializeLastSimpleProfilerWrite::apply(
    const gsl::not_null<std::atomic<double>*> last_write) {
  last_write->store(-std::numeric_limits<double>::infinity(),
                    std::memory_order_release);
}
}  // namespace profiling
