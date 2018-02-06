// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ErrorHandling/FloatingPointExceptions.hpp"

inline void setup_error_handling() {
  std::set_terminate([]() {
    Parallel::abort(
        "Terminate was called, calling Charm++'s abort function to properly "
        "terminate execution.");
  });
  enable_floating_point_exceptions();
}
