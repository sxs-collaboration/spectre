// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::FunctionsOfTime::FunctionOfTimeHelpers {
namespace ThreadsafeList_detail {
template <typename T>
struct Interval;
}  // namespace ThreadsafeList_detail

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
  using Interval = ThreadsafeList_detail::Interval<T>;

 public:
  /// No operations are valid on a default-initialized object except
  /// for assignment and deserialization.
  ThreadsafeList();
  ThreadsafeList(ThreadsafeList&& other);
  ThreadsafeList(const ThreadsafeList& other);
  ThreadsafeList& operator=(ThreadsafeList&& other);
  ThreadsafeList& operator=(const ThreadsafeList& other);
  ~ThreadsafeList();

  explicit ThreadsafeList(double initial_time);

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

  double initial_time() const;

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

    iterator(double initial_time, const Interval* interval);

    double initial_time_ = std::numeric_limits<double>::signaling_NaN();
    const Interval* interval_ = nullptr;
  };

  /// Iterate over all the intervals in the list, in reverse order
  /// (i.e., most recent first).
  /// @{
  iterator begin() const;
  iterator end() const;
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
bool operator==(const ThreadsafeList<T>& a, const ThreadsafeList<T>& b);

template <typename T>
bool operator!=(const ThreadsafeList<T>& a, const ThreadsafeList<T>& b);
}  // namespace domain::FunctionsOfTime::FunctionOfTimeHelpers
