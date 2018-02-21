// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <exception>

#include "Parallel/Abort.hpp"

inline void setup_error_handling() {
  std::set_terminate([]() {
    Parallel::abort(
        "Terminate was called, calling Charm++'s abort function to properly "
        "terminate execution.");
  });
}
