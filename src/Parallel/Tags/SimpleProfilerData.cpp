// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/Tags/SimpleProfilerData.hpp"

#include <cstddef>

#include "Utilities/System/ParallelInfo.hpp"

namespace Parallel::Tags {
SimpleProfilerData::type SimpleProfilerData::create_from_options() {
  return type{static_cast<size_t>(sys::procs_on_node(sys::my_node()))};
}
}  // namespace Parallel::Tags
