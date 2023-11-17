// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/System/Exit.hpp"

#include <charm++.h>
#include <exception>

namespace sys {

[[noreturn]] void exit(const int exit_code) {
  CkExit(exit_code);
  // the following call is never reached, but suppresses the warning that
  // a 'noreturn' function does return
  std::terminate();  // LCOV_EXCL_LINE
}

}  // namespace sys
