// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <exception>

namespace sys {

/// \ingroup SystemUtilitiesGroup
/// \brief Exit the program normally.
/// This should only be called once over all processors.
[[noreturn]] inline void exit() {
  CkExit();
  // the following call is never reached, but suppresses the warning that
  // a 'noreturn' function does return
  std::terminate();  // LCOV_EXCL_LINE
}

}  // namespace sys
