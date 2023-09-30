// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::FunctionsOfTime::FunctionOfTimeHelpers {
/// A list of time intervals that allows safe access to existing
/// elements even during modification.
///
/// Concurrent modification is not supported.  All non-modifying
/// operations except for serialization can be safely performed in
/// parallel with each other and with `insert` and will return a
/// consistent state.
template <typename T>
class ThreadsafeList {
 private:
  struct Interval {
    double expiration;
    T data;
    std::unique_ptr<Interval> previous;

    void pup(PUP::er& p);
  };

 public:
  /// No operations are valid on a default-initialized object except
  /// for assignment and deserialization.
  ThreadsafeList() = default;
  ThreadsafeList(ThreadsafeList&& other) { *this = std::move(other); }
  ThreadsafeList(const ThreadsafeList& other) { *this = other; }
  ThreadsafeList& operator=(ThreadsafeList&& other);
  ThreadsafeList& operator=(const ThreadsafeList& other);
  ~ThreadsafeList() = default;

  explicit ThreadsafeList(const double initial_time)
      : initial_time_(initial_time), most_recent_interval_(nullptr) {}

  /// Insert data valid over the interval from \p update_time to \p
  /// expiration_time.
  ///
  /// The \p update_time must be the same as the old expiration time.
  /// It is passed as a check that the calling code is computing the
  /// other arguments with the correct value.
  void insert(double update_time, T data, double expiration_time);

  struct IntervalInfo {
    double update;
    const T& data;
    double expiration;

    friend bool operator==(const IntervalInfo& a, const IntervalInfo& b) {
      return a.update == b.update and a.data == b.data and
             a.expiration == b.expiration;
    }
    friend bool operator!=(const IntervalInfo& a, const IntervalInfo& b) {
      return not(a == b);
    }
  };

  /// Obtain the start time, data, and expiration time for the time
  /// interval containing \p time.
  ///
  /// If \p time is at an interval boundary, the earlier interval is
  /// returned, except for the initial time, which returns the first
  /// interval.
  IntervalInfo operator()(double time) const;

  double initial_time() const { return initial_time_; }

  /// The expiration time of the last data interval.
  double expiration_time() const;

  /// The first interval boundary after \p time.  If \p time is an
  /// interval boundary, the one after it is returned.
  double expiration_after(double time) const;

  void pup(PUP::er& p);

  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = decltype(ThreadsafeList{}(double{}));
    using reference = value_type;
    using pointer = std::optional<value_type>;
    using difference_type = std::ptrdiff_t;

    iterator() = default;
    iterator& operator++();
    iterator operator++(int);
    reference operator*() const;
    pointer operator->() const;

    friend bool operator==(const iterator& a, const iterator& b) {
      return a.interval_ == b.interval_;
    }
    friend bool operator!=(const iterator& a, const iterator& b) {
      return not(a == b);
    }

   private:
    friend class ThreadsafeList;

    iterator(const double initial_time, const Interval* const interval)
        : initial_time_(initial_time), interval_(interval) {}

    double initial_time_ = std::numeric_limits<double>::signaling_NaN();
    const Interval* interval_ = nullptr;
  };

  /// Iterate over all the intervals in the list, in reverse order
  /// (i.e., most recent first).
  /// @{
  iterator begin() const;
  iterator end() const { return {}; }
  /// @}

 private:
  // interval_after_boundary controls behavior at boundary points.
  // True -> later interval, false -> earlier.  Initial time always
  // gives the first interval.
  const Interval& find_interval(double time,
                                bool interval_after_boundary) const;

  double initial_time_ = std::numeric_limits<double>::signaling_NaN();
  std::unique_ptr<Interval> interval_list_{};
  alignas(64) std::atomic<const Interval*> most_recent_interval_{};
  // Pad memory to avoid false-sharing when accessing most_recent_interval_
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  char unused_padding_[64 - (sizeof(std::atomic<const Interval*>) % 64)] = {};
};

template <typename T>
void ThreadsafeList<T>::Interval::pup(PUP::er& p) {
  p | expiration;
  p | data;
  p | previous;
}

template <typename T>
auto ThreadsafeList<T>::operator=(ThreadsafeList&& other) -> ThreadsafeList& {
  if (this == &other) {
    return *this;
  }
  initial_time_ = other.initial_time_;
  interval_list_ = std::move(other.interval_list_);
  most_recent_interval_.store(interval_list_.get(), std::memory_order_release);
  other.most_recent_interval_.store(nullptr, std::memory_order_release);
  return *this;
}

template <typename T>
auto ThreadsafeList<T>::operator=(const ThreadsafeList& other)
    -> ThreadsafeList& {
  if (this == &other) {
    return *this;
  }
  initial_time_ = other.initial_time_;

  std::unique_ptr<Interval>* previous_pointer = &interval_list_;
  for (auto&& entry : other) {
    // make_unique doesn't work on aggregates until C++20
    previous_pointer->reset(
        new Interval{entry.expiration, entry.data, nullptr});
    previous_pointer = &(*previous_pointer)->previous;
  }

  most_recent_interval_.store(interval_list_.get(), std::memory_order_release);
  return *this;
}

template <typename T>
void ThreadsafeList<T>::insert(const double update_time, T data,
                               const double expiration_time) {
  const auto* old_interval =
      most_recent_interval_.load(std::memory_order_acquire);
  const double old_expiration =
      old_interval != nullptr ? old_interval->expiration : initial_time_;
  if (old_expiration != update_time) {
    ERROR("Tried to insert at time "
          << update_time << ", which is not the old expiration time "
          << old_expiration);
  }
  if (expiration_time <= update_time) {
    ERROR("Expiration time " << expiration_time << " is not after update time "
                             << update_time);
  }
  // make_unique doesn't work on aggregates until C++20
  std::unique_ptr<Interval> new_interval(new Interval{
      expiration_time, std::move(data), std::move(interval_list_)});
  const auto* const new_interval_p = new_interval.get();
  interval_list_ = std::move(new_interval);
  if (not most_recent_interval_.compare_exchange_strong(
          old_interval, new_interval_p, std::memory_order_acq_rel)) {
    ERROR("Attempt at concurrent modification detected.");
  }
}

template <typename T>
auto ThreadsafeList<T>::operator()(const double time) const -> IntervalInfo {
  if (time < initial_time_ and not equal_within_roundoff(time, initial_time_)) {
    ERROR("Requested time " << time << " precedes earliest time "
                            << initial_time_);
  }
  const auto& interval = find_interval(time, false);
  const double start = interval.previous != nullptr
                           ? interval.previous->expiration
                           : initial_time_;
  return {start, interval.data, interval.expiration};
}

template <typename T>
double ThreadsafeList<T>::expiration_time() const {
  auto* interval = most_recent_interval_.load(std::memory_order_acquire);
  return interval != nullptr ? interval->expiration : initial_time_;
}

template <typename T>
double ThreadsafeList<T>::expiration_after(const double time) const {
  return find_interval(time, true).expiration;
}

template <typename T>
void ThreadsafeList<T>::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.

  if (version != 0) {
    ERROR("Unrecognized version " << version);
  }

  p | initial_time_;
  // We can't just `p | interval_list_` because serialization has to
  // be threadsafe.  (Deserialization does not.)
  if (p.isUnpacking()) {
    bool empty{};
    p | empty;
    interval_list_.reset();
    if (not empty) {
      interval_list_ = std::make_unique<Interval>();
      p | *interval_list_;
    }
    most_recent_interval_.store(interval_list_.get(),
                                std::memory_order_release);
  } else {
    const Interval* const threadsafe_interval_list =
        most_recent_interval_.load(std::memory_order_acquire);
    bool empty = threadsafe_interval_list == nullptr;
    p | empty;
    if (not empty) {
      // const_cast is fine both because (a) pupping only modifies the
      // object if `p.isUnpacking()`, which is false here, and (b) we
      // know this points into mutable storage as part of
      // interval_list_.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      p | const_cast<Interval&>(*threadsafe_interval_list);
    }
  }
}

template <typename T>
auto ThreadsafeList<T>::iterator::operator++() -> iterator& {
  interval_ = interval_->previous.get();
  return *this;
}

template <typename T>
auto ThreadsafeList<T>::iterator::operator++(int) -> iterator {
  auto result = *this;
  ++*this;
  return result;
}

template <typename T>
auto ThreadsafeList<T>::iterator::operator*() const -> reference {
  return {interval_->previous != nullptr ? interval_->previous->expiration
                                         : initial_time_,
          interval_->data, interval_->expiration};
}

template <typename T>
auto ThreadsafeList<T>::iterator::operator->() const -> pointer {
  return {**this};
}

template <typename T>
auto ThreadsafeList<T>::begin() const -> iterator {
  return {initial_time_, most_recent_interval_.load(std::memory_order_acquire)};
}

template <typename T>
auto ThreadsafeList<T>::find_interval(const double time,
                                      const bool interval_after_boundary) const
    -> const Interval& {
  auto* interval = most_recent_interval_.load(std::memory_order_acquire);
  if (interval == nullptr) {
    ERROR("Attempt to access an empty function of time.");
  }
  if (time > interval->expiration or
      (interval_after_boundary and time == interval->expiration)) {
    ERROR("Attempt to evaluate at time "
          << time << ", which is after the expiration time "
          << interval->expiration);
  }
  // Loop over the intervals until we find the one containing `time`,
  // possibly at the endpoint determined by `interval_after_boundary`.
  for (;;) {
    const auto* const previous_interval = interval->previous.get();
    if (previous_interval == nullptr or time > previous_interval->expiration or
        (interval_after_boundary and time == previous_interval->expiration)) {
      return *interval;
    }
    interval = previous_interval;
  }
}

template <typename T>
bool operator==(const ThreadsafeList<T>& a, const ThreadsafeList<T>& b) {
  if (a.initial_time() != b.initial_time()) {
    return false;
  }
  auto a_iter = a.begin();
  auto b_iter = b.begin();
  while (a_iter != a.end() and b_iter != b.end()) {
    if (*a_iter++ != *b_iter++) {
      return false;
    }
  }
  return a_iter == a.end() and b_iter == b.end();
}

template <typename T>
bool operator!=(const ThreadsafeList<T>& a, const ThreadsafeList<T>& b) {
  return not(a == b);
}
}  // namespace domain::FunctionsOfTime::FunctionOfTimeHelpers
