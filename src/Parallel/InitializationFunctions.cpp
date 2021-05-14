// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/InitializationFunctions.hpp"

#include <exception>

#include "Utilities/ErrorHandling/Error.hpp"

void setup_error_handling() {
  std::set_terminate([]() {
    std::exception_ptr exception = std::current_exception();
    if (exception) {
      try {
        std::rethrow_exception(exception);
      } catch (std::exception& ex) {
        ERROR("Terminated due to an uncaught exception: " << ex.what());
      } catch (...) {
        ERROR("Terminated due to unknown exception\n");
      }
    } else {
      ERROR(
          "Terminate was called for an unknown reason (not an uncaught "
          "exception), calling Charm++'s abort function to properly "
          "terminate execution.");
    }
  });
}
