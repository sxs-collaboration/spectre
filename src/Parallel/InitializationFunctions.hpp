// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ErrorHandling/FloatingPointExceptions.hpp"

inline void setup_error_handling() {
  std::set_terminate(
      []() { Parallel::abort("Called terminate. Aborting..."); });
  enable_floating_point_exceptions();
}
