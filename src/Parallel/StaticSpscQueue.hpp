// Distributed under the MIT License.
// See LICENSE.txt for details.

/*
 *The original code is distributed under the following copyright and license:
 *
 * Copyright (c) 2020 Erik Rigtorp <erik@rigtorp.se>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 * SXS Modifications:
 * 1. Casing to match SpECTRE conventions
 * 2. Static capacity
 * 3. Storage is std::array
 * 4. Switch to west-const
 *
 */

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <new>  // Placement new
#include <stdexcept>
#include <type_traits>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Requires.hpp"

namespace Parallel {
/*!
 * \brief A static capacity runtime-sized single-producer single-consumer
 * lockfree queue.
 *
 * As long as only one thread reads and writes simultaneously the queue is
 * threadsafe. Which threads read and write can change throughout program
 * execution, the important thing is that there is no instance during the
 * execution where more than one thread tries to read and where more than one
 * thread tries to write.
 *
 * \note This class is intentionally not serializable since handling
 * threadsafety around serialization requires careful thought of the individual
 * circumstances.
 */
template <typename T, size_t Capacity>
class StaticSpscQueue {
 private:
#ifdef __cpp_lib_hardware_interference_size
  static constexpr size_t cache_line_size_ =
      std::hardware_destructive_interference_size;
#else
  static constexpr size_t cache_line_size_ = 64;
#endif

  // Padding to avoid false sharing between slots_ and adjacent allocations
  static constexpr size_t padding_ = (cache_line_size_ - 1) / sizeof(T) + 1;

 public:
  StaticSpscQueue() = default;
  ~StaticSpscQueue() {
    // Destruct objects in the buffer.
    while (front()) {
      pop();
    }
  }

  StaticSpscQueue(const StaticSpscQueue&) = delete;
  StaticSpscQueue& operator=(const StaticSpscQueue&) = delete;
  StaticSpscQueue(StaticSpscQueue&&) = delete;
  StaticSpscQueue& operator=(StaticSpscQueue&&) = delete;

  /// Construct a new element at the end of the queue in place.
  ///
  /// Uses placement new for in-place construction.
  ///
  /// \warning This may overwrite existing elements if `capacity()` is
  /// exceeded without warning.
  template <typename... Args>
  void emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args&&...>) {
    static_assert(std::is_constructible_v<T, Args&&...>,
                  "T must be constructible with Args&&...");
    const auto write_index = write_index_.load(std::memory_order_relaxed);
    auto next_write_index = write_index + 1;
    if (next_write_index == capacity_) {
      next_write_index = 0;
    }
    while (next_write_index == read_index_cache_) {
      read_index_cache_ = read_index_.load(std::memory_order_acquire);
    }
    new (&data_[write_index + padding_]) T(std::forward<Args>(args)...);
    write_index_.store(next_write_index, std::memory_order_release);
  }

  /// Construct a new element at the end of the queue in place.
  ///
  /// Uses placement new for in-place construction.
  ///
  /// Returns `true` if the emplacement succeeded and `false` if it did
  /// not. If it failed then the queue is currently full.
  template <typename... Args>
  [[nodiscard]] bool try_emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<T, Args&&...>) {
    static_assert(std::is_constructible_v<T, Args&&...>,
                  "T must be constructible with Args&&...");
    const auto write_index = write_index_.load(std::memory_order_relaxed);
    auto next_write_index = write_index + 1;
    if (next_write_index == capacity_) {
      next_write_index = 0;
    }
    if (next_write_index == read_index_cache_) {
      read_index_cache_ = read_index_.load(std::memory_order_acquire);
      if (next_write_index == read_index_cache_) {
        return false;
      }
    }
    new (&data_[write_index + padding_]) T(std::forward<Args>(args)...);
    write_index_.store(next_write_index, std::memory_order_release);
    return true;
  }

  /// Push a new element to the end of the queue.
  ///
  /// Uses `emplace()` internally.
  ///
  /// \warning This may overwrite existing elements if `capacity()` is
  /// exceeded without warning.
  void push(const T& v) noexcept(std::is_nothrow_copy_constructible_v<T>) {
    static_assert(std::is_copy_constructible_v<T>,
                  "T must be copy constructible");
    emplace(v);
  }

  /// Push a new element to the end of the queue.
  ///
  /// Uses `emplace()` internally.
  ///
  /// \warning This may overwrite existing elements if `capacity()` is
  /// exceeded without warning.
  template <typename P, Requires<std::is_constructible_v<T, P&&>> = nullptr>
  void push(P&& v) noexcept(std::is_nothrow_constructible_v<T, P&&>) {
    emplace(std::forward<P>(v));
  }

  /// Push a new element to the end of the queue. Returns `false` if the queue
  /// is at capacity and does not push the new object, otherwise returns `true`.
  ///
  /// Uses `try_emplace()` internally.
  [[nodiscard]] bool try_push(const T& v) noexcept(
      std::is_nothrow_copy_constructible_v<T>) {
    static_assert(std::is_copy_constructible_v<T>,
                  "T must be copy constructible");
    return try_emplace(v);
  }

  /// Push a new element to the end of the queue. Returns `false` if the queue
  /// is at capacity and does not push the new object, otherwise returns `true`.
  ///
  /// Uses `try_emplace()` internally.
  template <typename P, Requires<std::is_constructible_v<T, P&&>> = nullptr>
  [[nodiscard]] bool try_push(P&& v) noexcept(
      std::is_nothrow_constructible_v<T, P&&>) {
    return try_emplace(std::forward<P>(v));
  }

  /// Returns the first element from the queue.
  ///
  /// \note Returns `nullptr` if the queue is empty.
  [[nodiscard]] T* front() noexcept {
    const auto read_index = read_index_.load(std::memory_order_relaxed);
    if (read_index == write_index_cache_) {
      write_index_cache_ = write_index_.load(std::memory_order_acquire);
      if (write_index_cache_ == read_index) {
        return nullptr;
      }
    }
    return &data_[read_index + padding_];
  }

  /// Removes the first element from the queue.
  void pop() {
    static_assert(std::is_nothrow_destructible_v<T>,
                  "T must be nothrow destructible");
    const auto read_index = read_index_.load(std::memory_order_relaxed);
#ifdef SPECTRE_DEBUG
    const auto write_index = write_index_.load(std::memory_order_acquire);
    ASSERT(write_index != read_index,
           "Can't pop an element from an empty queue. read_index: "
               << read_index << " write_index " << write_index);
#endif  // SPECTRE_DEBUG
    data_[read_index + padding_].~T();
    auto next_read_index = read_index + 1;
    if (next_read_index == capacity_) {
      next_read_index = 0;
    }
    if (read_index == write_index_cache_) {
      write_index_cache_ = next_read_index;
    }
    read_index_.store(next_read_index, std::memory_order_release);
  }

  /// Returns the size of the queue at a particular hardware state.
  ///
  /// Note that while this can be checked in a threadsafe manner, it is up to
  /// the user to guarantee that another thread does not change the queue
  /// between when `size()` is called and how the result is used.
  [[nodiscard]] size_t size() const noexcept {
    std::ptrdiff_t diff = static_cast<std::ptrdiff_t>(
                              write_index_.load(std::memory_order_acquire)) -
                          static_cast<std::ptrdiff_t>(
                              read_index_.load(std::memory_order_acquire));
    if (diff < 0) {
      diff += static_cast<std::ptrdiff_t>(capacity_);
    }
    return static_cast<size_t>(diff);
  }

  /// Returns `true` if the queue may be empty, otherwise `false`.
  ///
  /// Note that while this can be checked in a threadsafe manner, it is up to
  /// the user to guarantee that another thread does not change the queue
  /// between when `empty()` is called and how the result is used.
  [[nodiscard]] bool empty() const noexcept {
    return write_index_.load(std::memory_order_acquire) ==
           read_index_.load(std::memory_order_acquire);
  }

  /// Returns the capacity of the queue.
  [[nodiscard]] size_t capacity() const noexcept { return capacity_ - 1; }

 private:
  static constexpr size_t capacity_ = Capacity + 1;
  std::array<T, capacity_ + 2 * padding_> data_{};

  // Align to cache line size in order to avoid false sharing
  // read_index_cache_ and write_index_cache_ is used to reduce the amount of
  // cache coherency traffic
  alignas(cache_line_size_) std::atomic<size_t> write_index_{0};
  alignas(cache_line_size_) size_t read_index_cache_{0};
  alignas(cache_line_size_) std::atomic<size_t> read_index_{0};
  alignas(cache_line_size_) size_t write_index_cache_{0};

  // Padding to avoid adjacent allocations from sharing a cache line with
  // write_index_cache_
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  char padding_data_[cache_line_size_ - sizeof(write_index_cache_)]{};
};
}  // namespace Parallel
