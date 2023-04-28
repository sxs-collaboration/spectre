// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/InitializationFunctions.hpp"

#include <charm++.h>
#include <exception>
#include <sstream>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/System/Abort.hpp"

void setup_error_handling() {
  std::set_terminate([]() {
    std::exception_ptr exception = std::current_exception();
    if (exception) {
      try {
        std::rethrow_exception(exception);
      } catch (std::exception& ex) {
        std::ostringstream os;
        os << "Terminated due to an uncaught exception:\n" << ex.what();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        CkError("%s\n", os.str().c_str());
      } catch (...) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        CkError("Terminated due to unknown exception\n");
      }
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      CkError(
          "Terminate was called for an unknown reason (not an uncaught "
          "exception), calling Charm++'s abort function to properly "
          "terminate execution.");
    }
    sys::abort("");
  });
}
