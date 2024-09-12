// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/System/AttachDebugger.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <unistd.h>

namespace sys {
void attach_debugger() {
  const char* env_enable_parallel_debug =
      // NOLINTNEXTLINE(concurrency-mt-unsafe)
      std::getenv("SPECTRE_ATTACH_DEBUGGER");
  if (env_enable_parallel_debug != nullptr) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    char hostname[2048];
    gethostname(static_cast<char*>(hostname), sizeof(hostname));

    const std::string output_info =
        std::string{"pid:"} + std::to_string(getpid()) + std::string{":host:"} +
        std::string{static_cast<char*>(hostname)} + "\n";
    std::cout << output_info << std::flush;
    // NOLINTNEXTLINE(misc-const-correctness)
    volatile int i = 10;
    while (i == 10) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(std::chrono::seconds{i});
    }
  }
}
}  // namespace sys
