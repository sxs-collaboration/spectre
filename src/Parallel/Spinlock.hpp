// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>

namespace Parallel {
/*!
 * \brief A simple spinlock implemented in `std::atomic`s
 *
 * Implementation adapted from https://rigtorp.se/spinlock/
 */
class Spinlock {
 public:
  Spinlock() = default;
  Spinlock(const Spinlock&) = delete;
  Spinlock& operator=(const Spinlock&) = delete;
  Spinlock(Spinlock&&) = delete;
  Spinlock& operator=(Spinlock&&) = delete;
  ~Spinlock() = default;

  /// \brief Acquire the lock. Will block until the lock is acquired.
  void lock() {
    for (;;) {
      // Optimistically assume the lock is free on the first try
      if (not lock_.exchange(true, std::memory_order_acquire)) {
        return;
      }
      // Wait for lock to be released without generating cache misses
      while (lock_.load(std::memory_order_relaxed)) {
        // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
        // hyper-threads
        //
        // If no hyperthreading is being used, this will actually slow down the
        // code.
        //
        // We keep this comment and code around just in case we want to
        // experiment with it in the future.
        //
        // __builtin_ia32_pause();
      }
    }
  }

  /// \brief Try to acquire the lock.
  ///
  /// Returns `true` if the lock was acquired, `false` if it wasn't.
  bool try_lock() {
    // First do a relaxed load to check if lock is free in order to prevent
    // unnecessary cache misses if someone does while(!try_lock())
    return not lock_.load(std::memory_order_relaxed) and
           not lock_.exchange(true, std::memory_order_acquire);
  }

  /// \brief Release the lock.
  void unlock() { lock_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> lock_{false};
};
}  // namespace Parallel
