// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <limits>
#include <new>  // for hardware_destructive_interference_size

namespace Parallel {
/*!
 * \brief A two-state spinlock that allows multiple readers of a shared resource
 * to acquire the lock simultaneously.
 *
 * A spinlock (i.e. a non-yielding lock) that can be used to guard a resource
 * where multiple threads can safely perform some types of operations
 * (e.g. reader entries from a container), while only one thread can safely
 * perform other types of operations (e.g. modifying the container).
 *
 * ### Implementation
 * We lock by using a `std::atomic<std::int64_t>` (a signed integer). The lock
 * is not locked if the integer is `0`. It is read-locked if it is positive
 * and write-locked if it is negative. To read-lock we first check that the
 * integer is non-negative and then `fetch_add(1, std::memory_order_acq_rel)`,
 * then check that the number _before_ the increment is non-negative. The
 * reason for the second check is that between checking that the lock is not
 * write-locked and the `fetch_add()` operation, it could be write-locked by
 * another thread. To write-lock, we check that the integer is `0` and if so
 * we set it to `std::numeric_limits<std::int64_t>::lowest()` (i.e. `-2^{63}`).
 * We achieve this by using a `compare_exchange_strong()` with failure memory
 * order `relaxed` and success memory order `acq_rel`. This guarantees that
 * different threads synchronize using the lock.
 *
 * A thread read-unlocks the lock using a
 * `fetch_sub(1, std::memory_order_acq_rel)` operation.
 *
 * A thread write-unlocks the lock using a `store(0, std::memory_order_release)`
 * operation.
 */
class MultiReaderSpinlock {
 public:
  MultiReaderSpinlock() = default;
  MultiReaderSpinlock(const MultiReaderSpinlock&) = delete;
  MultiReaderSpinlock& operator=(const MultiReaderSpinlock&) = delete;
  MultiReaderSpinlock(MultiReaderSpinlock&&) = delete;
  MultiReaderSpinlock& operator=(MultiReaderSpinlock&&) = delete;
  ~MultiReaderSpinlock() = default;

  /// \brief Acquire the lock in a reader state.
  ///
  /// \note Multiple threads can acquire a reader state simultaneously.
  void read_lock() noexcept {
    for (;;) {
      while (lock_.load(std::memory_order_relaxed) < 0) {
      }

      if (lock_.fetch_add(1, std::memory_order_acquire) > -1) {
        return;
      }
    }
  }

  /// \brief Release the lock from a reader state.
  ///
  /// \note Since multiple threads can simultaneously acquire a reader state,
  /// unlock from a single reader does not guarantee a writer can acquire the
  /// lock in a write state.
  void read_unlock() noexcept { lock_.fetch_sub(1, std::memory_order_release); }

  /// \brief Acquire the lock in a writer state.
  ///
  /// Once acquired, no other thread can acquire the lock in either a read or
  /// a write state until this thread unlocks it.
  void write_lock() noexcept {
    for (;;) {
      std::int64_t current_value{0};
      if (lock_.compare_exchange_strong(
              current_value, std::numeric_limits<std::int64_t>::lowest(),
              std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
      }

      while (lock_.load(std::memory_order_relaxed) != 0) {
      }
    }
  }

  /// \brief Release the lock from a reader state.
  void write_unlock() noexcept { lock_.store(0, std::memory_order_release); }

 private:
#ifdef __cpp_lib_hardware_interference_size
  static constexpr size_t cache_line_size_ =
      std::hardware_destructive_interference_size;
#else
  static constexpr size_t cache_line_size_ = 64;
#endif

  alignas(cache_line_size_) std::atomic<std::int64_t> lock_{0};
  // Ensure we pad the end to avoid false sharing. Unlikely to happen because
  // of the layout, but better to be safe.
#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-private-field"
#endif
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  char padding_[cache_line_size_ - sizeof(std::atomic<std::int64_t>)] = {};
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif
};
}  // namespace Parallel
