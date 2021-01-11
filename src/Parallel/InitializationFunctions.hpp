// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <exception>

#include "Utilities/ErrorHandling/Error.hpp"

inline void setup_error_handling() {
  std::set_terminate([]() {
    ERROR(
        "Terminate was called, calling Charm++'s abort function to properly "
        "terminate execution.");
  });
}
