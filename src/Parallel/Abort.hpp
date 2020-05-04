// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function Parallel::abort.

#pragma once

#include <charm++.h>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <string>

namespace Parallel {

/// \ingroup ParallelGroup
/// Abort the program with an error message.
///
/// \details This function calls CkExit with a non-zero argument to indicate a
/// failure, unless the SPECTRE_TRAP_ON_ERROR environmental variable is set, in
/// which case it raises SIGTRAP.
[[noreturn]] inline void abort(const std::string& message) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
  CkPrintf("%s\n", message.c_str());
#pragma GCC diagnostic pop
  char* const trap_for_debugger = std::getenv("SPECTRE_TRAP_ON_ERROR");
  if (trap_for_debugger != nullptr) {
    std::raise(SIGTRAP);
  }
  CkExit(1);
  // the following call is never reached, but suppresses the warning that
  // a 'noreturn' function does return
  std::terminate();  // LCOV_EXCL_LINE
}

}  // namespace Parallel
