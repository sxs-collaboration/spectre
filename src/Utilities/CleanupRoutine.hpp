// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

/// An object that calls a functor when it goes out of scope.
///
/// \snippet Test_CleanupRoutine.cpp cleanup_routine
template <typename F>
class CleanupRoutine {
 public:
  CleanupRoutine(F routine) : routine_(std::move(routine)) {}

  // Copying/moving don't make sense, and it's not clear what a
  // moved-out-of instance would be expected to do.
  CleanupRoutine(const CleanupRoutine&) = delete;
  CleanupRoutine(CleanupRoutine&&) = delete;
  CleanupRoutine& operator=(const CleanupRoutine&) = delete;
  CleanupRoutine& operator=(CleanupRoutine&&) = delete;

  ~CleanupRoutine() { routine_(); }

 private:
  F routine_;
};
