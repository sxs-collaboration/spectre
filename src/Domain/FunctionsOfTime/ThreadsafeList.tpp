// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/FunctionsOfTime/ThreadsafeList.hpp"

#include <atomic>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::FunctionsOfTime::FunctionOfTimeHelpers {
namespace ThreadsafeList_detail {
template <typename T>
struct Interval {
  double expiration;
  T data;
  std::unique_ptr<Interval> previous;

  void pup(PUP::er& p);
};

template <typename T>
void Interval<T>::pup(PUP::er& p) {
  p | expiration;
  p | data;
  p | previous;
}
}  // namespace ThreadsafeList_detail

template <typename T>
ThreadsafeList<T>::ThreadsafeList() = default;

template <typename T>
ThreadsafeList<T>::ThreadsafeList(ThreadsafeList&& other) {
  *this = std::move(other);
}

template <typename T>
ThreadsafeList<T>::ThreadsafeList(const ThreadsafeList& other) {
  *this = other;
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
ThreadsafeList<T>::~ThreadsafeList() = default;

template <typename T>
ThreadsafeList<T>::ThreadsafeList(const double initial_time)
    : initial_time_(initial_time), most_recent_interval_(nullptr) {}

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
double ThreadsafeList<T>::initial_time() const {
  return initial_time_;
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
ThreadsafeList<T>::iterator::iterator(const double initial_time,
                                      const Interval* const interval)
    : initial_time_(initial_time), interval_(interval) {}

template <typename T>
auto ThreadsafeList<T>::begin() const -> iterator {
  return {initial_time_, most_recent_interval_.load(std::memory_order_acquire)};
}

template <typename T>
auto ThreadsafeList<T>::end() const -> iterator {
  return {};
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
      (interval_after_boundary and time == interval->expiration and
       interval->expiration < std::numeric_limits<double>::infinity())) {
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
