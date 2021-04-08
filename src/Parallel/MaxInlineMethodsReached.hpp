// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

namespace Parallel::detail {

// Allow 64 inline entry method calls before we fall back to Charm++. This is
// done to avoid blowing the stack.
inline bool max_inline_entry_methods_reached() noexcept {
  thread_local size_t approx_stack_depth = 0;
#ifndef SPECTRE_PROFILING
  approx_stack_depth++;
  if (approx_stack_depth < 64) {
    return false;
  }
  approx_stack_depth = 0;
#endif
  return true;
}
}  // namespace Parallel::detail
