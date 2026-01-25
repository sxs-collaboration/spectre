// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>  // for hardware_destructive_interference_size
#include <stdexcept>
#include <utility>
#include <vector>

#include "Parallel/MultiReaderSpinlock.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace Parallel {
/*!
 * \brief A threadsafe parallel first-in-first-out cache.
 */
template <class T>
class FifoCache {
  using stored_type = std::pair<MultiReaderSpinlock, T>;
  using value_type = std::vector<stored_type>;

 public:
  /*!
   * \brief Wrapper type used as the result from `find` to ensure correct
   * thread safety.
   *
   * Use `.value()` to get the stored value.
   *
   * Use `.has_value()` to check if a value is stored.
   */
  struct Cached {
   public:
    Cached() = delete;
    explicit Cached(const stored_type* t);
    /// NOLINTNEXTLINE(google-explicit-constructor)
    Cached(const std::nullopt_t /*unused*/) : Cached{nullptr} {}
    Cached(const Cached& rhs);
    Cached& operator=(const Cached& rhs);
    Cached(Cached&& rhs) noexcept(true);
    Cached& operator=(Cached&& rhs) noexcept(true);
    ~Cached() noexcept(true);

    /// \brief Returns a reference to the held object.
    ///
    /// \throws std::runtime_error if no value
    auto value() const -> const T&;

    /// \brief Returns `true` if a value is stored, otherwise returns `false`.
    bool has_value() const { return t_ != nullptr; }

   private:
    stored_type* t_;
  };

  /// \brief Create a FifoCache that has \p capacity.
  explicit FifoCache(std::unsigned_integral auto capacity);

  FifoCache() = delete;
  FifoCache(const FifoCache& rhs) = delete;
  FifoCache& operator=(const FifoCache& rhs) = delete;
  FifoCache(FifoCache&& rhs) = delete;
  FifoCache& operator=(FifoCache&& rhs) = delete;
  ~FifoCache() = default;

  /*!
   * \brief Pushes the entry computed by `compute_value()` to the front of the
   * queue, ejecting the last entry if the capacity is reached. The inserted
   * entry is returned.
   *
   * This function allows lazy computation of the value, which means the
   * computation is elided if another thread pushes the new cache entry before
   * this one.
   *
   * The predicate must satisfy `predicate(t) == true` to avoid inserting
   * duplicates. This is best guaranteed by passing the same predicate that
   * would be passed to calls to `find()`.
   */
  template <class ComputeValue, class UnaryPredicate>
  auto push(ComputeValue&& compute_value, const UnaryPredicate& predicate)
      -> Cached;

  /*!
   * \brief Pushes the entry `t` to the front of the queue,
   * ejecting the last entry if the capacity is reached. The inserted
   * entry is returned.
   *
   * The predicate must satisfy `predicate(t) == true` to avoid inserting
   * duplicates. This is best guaranteed by passing the same predicate that
   * would be passed to calls to `find()`.
   */
  template <class UnaryPredicate>
  auto push(T t, const UnaryPredicate& predicate) -> Cached;

  /*!
   * \brief Get the first element that matches `predicate`.
   *
   * If no value in the cache matches the predicate then `result.has_value()`
   * is `false` and `result.value()` will throw.
   *
   * The return type is designed to handle the locking and unlocking
   * of the data to ensure thread safety.
   */
  template <class UnaryPredicate>
  [[nodiscard]] auto find(const UnaryPredicate& predicate) const -> Cached;

 private:
#if defined(__cpp_lib_hardware_interference_size)
  static constexpr size_t cache_line_size_ =
      std::hardware_destructive_interference_size;
#else
  static constexpr size_t cache_line_size_ = 64;
#endif

  // std::vector does not have an atomic size so in order to prevent tearing
  // we have to track the size separately.
  alignas(cache_line_size_) std::atomic<std::uint64_t> size_{0};
  alignas(cache_line_size_) std::mutex write_lock_{};
  alignas(cache_line_size_) value_type data_{};
  // Ensure we pad the end to avoid false sharing. Unlikely to happen because
  // of the layout, but better to be safe.
#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-private-field"
#endif
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  char padding_[cache_line_size_ - sizeof(value_type) % cache_line_size_] = {};
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif
};

template <class T>
FifoCache<T>::FifoCache(const std::unsigned_integral auto capacity)
    : data_(capacity) {
  ASSERT(capacity > 0, "Must have a positive capacity but got " << capacity);
}

template <class T>
template <class ComputeValue, class UnaryPredicate>
auto FifoCache<T>::push(ComputeValue&& compute_value,
                        const UnaryPredicate& predicate) -> Cached {
  std::lock_guard guard(write_lock_);
  auto size = size_.load(std::memory_order_acquire);
  for (size_t i = 0; i < size; ++i) {
    Cached vt{std::addressof(data_[i])};
    if (predicate(vt.value())) {
      return {vt};
    }
  }

  // Compute the new value _before_ locking data to minimize locked time.
  T new_value = std::forward<ComputeValue>(compute_value)();
  if (size < data_.capacity()) {
    ++size;
  }
  data_[size - 1].first.write_lock();
  for (size_t i = size - 1; i > 0; --i) {
    data_[i - 1].first.write_lock();
    data_[i].second = std::move(data_[i - 1].second);
  }
  data_[0].second = std::move(new_value);
  for (size_t i = 0; i < size; ++i) {
    data_[i].first.write_unlock();
  }
  size_.store(size, std::memory_order_release);
  return find(predicate);
}

template <class T>
template <class UnaryPredicate>
auto FifoCache<T>::push(T t, const UnaryPredicate& predicate) -> Cached {
  return push(
      [t_local = std::move(t)]() mutable -> T  // NOLINT(spectre-mutable)
      { return std::move(t_local); },
      predicate);
}

template <class T>
template <class UnaryPredicate>
[[nodiscard]] auto FifoCache<T>::find(const UnaryPredicate& predicate) const
    -> Cached {
  const auto size = size_.load(std::memory_order_relaxed);
  for (size_t i = 0; i < size; ++i) {
    Cached vt{std::addressof(data_[i])};
    if (predicate(vt.value())) {
      return vt;
    }
  }
  return Cached{nullptr};
}

template <class T>
FifoCache<T>::Cached::Cached(const stored_type* t)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    : t_(const_cast<stored_type*>(t)) {
  if (t_ != nullptr) {
    t_->first.read_lock();
  }
}

template <class T>
FifoCache<T>::Cached::Cached(const Cached& rhs) : t_(rhs.t_) {
  if (t_ != nullptr) {
    t_->first.read_lock();
  }
}

template <class T>
typename FifoCache<T>::Cached& FifoCache<T>::Cached::operator=(
    const Cached& rhs) {
  if (&rhs == this) {
    return *this;
  }
  if (t_ != nullptr) {
    t_->first.read_unlock();
  }
  t_ = rhs.t_;
  if (t_ != nullptr) {
    t_->first.read_lock();
  }
  return *this;
}

template <class T>
FifoCache<T>::Cached::Cached(Cached&& rhs) noexcept(true) : t_(rhs.t_) {
  rhs.t_ = nullptr;
}

template <class T>
typename FifoCache<T>::Cached& FifoCache<T>::Cached::operator=(
    Cached&& rhs) noexcept(true) {
  if (&rhs == this) {
    return *this;
  }
  if (t_ != nullptr) {
    t_->first.read_unlock();
  }
  t_ = rhs.t_;
  rhs.t_ = nullptr;
  return *this;
}

template <class T>
FifoCache<T>::Cached::~Cached() noexcept(true) {
  if (t_ != nullptr) {
    t_->first.read_unlock();
  }
}

template <class T>
auto FifoCache<T>::Cached::value() const -> const T& {
  if (UNLIKELY(not has_value())) {
    throw std::runtime_error{"No value in FifoCache."};
  }
  return t_->second;
}
}  // namespace Parallel
