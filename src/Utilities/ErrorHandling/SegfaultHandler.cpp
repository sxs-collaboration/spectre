// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

#include "Utilities/ErrorHandling/Error.hpp"

#include <csignal>

namespace {
[[noreturn]] void segfault_signal_handler(int /*signal*/) {
  ERROR("Segmentation fault!");
}
}  // namespace

void enable_segfault_handler() {
  std::signal(SIGSEGV, segfault_signal_handler);
}
