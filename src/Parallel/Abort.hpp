// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function Parallel::abort.

#pragma once

#include <charm++.h>
#include <exception>
#include <string>

namespace Parallel {

/// \ingroup ParallelGroup
/// Abort the program with an error message.
[[noreturn]] inline void abort(const std::string& message) {
  CkAbort(message.c_str());
  // the following call is never reached, but suppresses the warning that
  // a 'noreturn' functions does return
  std::terminate();  // LCOV_EXCL_LINE
}

}  // namespace Parallel
