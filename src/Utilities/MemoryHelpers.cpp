// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/MemoryHelpers.hpp"

#include <new>

#include "ErrorHandling/Error.hpp"

namespace {
[[noreturn]] void report_failure() noexcept {
  // Make sure we don't go into an infinite loop if we fail to
  // allocate in this function.
  std::set_new_handler(nullptr);

  ERROR("Failed to allocate memory.");
}
}  // namespace


void setup_memory_allocation_failure_reporting() noexcept {
  std::set_new_handler(report_failure);
}
