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

  /// Remove the oldest data in the list, leaving at least \p length
  /// entries.
  ///
  /// If the list is truncated to zero length, the caller must ensure
  /// that no concurrent access or modification is performed.  The
  /// list is guaranteed to be empty afterwards.
  ///
  /// Otherwise, it is the caller's responsibility to ensure that
  /// there are no concurrent truncations and that, during this call,
  /// no reader attempts to look up what will become the new oldest
  /// interval.  Storing references to that interval's data is safe,
  /// and lookups of that interval are safe if they are sequenced
  /// after this call.  Concurrent insertions are safe.
  ///
  /// If the list is initially shorter than \p length, it is left
  /// unchanged.  Otherwise, if concurrent insertions occur, the
  /// amount of removed data is approximate, but will not leave the
  /// list shorter than the requested value.  In the absence of
  /// concurrent modifications, the list will be left with exactly the
  /// requested length.
  void truncate_to_length(size_t length);

  /// Remove the oldest data in the list, retaining access to data
  /// back to at least the interval containing \p time.
  ///
  /// If \p time is before `initial_time()`, the list is not modified.
  /// If \p time is after `expiration_time()`, the amount of removed
  /// data is unspecified.
  ///
  /// The amount of removed data is approximate.  There may be
  /// slightly more data remaining after a call than requested.
  ///
  /// It is the caller's responsibility to ensure that there are no
  /// concurrent truncations.  Concurrent insertions are safe.
  void truncate_at_time(double time);

  /// Empty the list.  Equivalent to `truncate_to_length(0)`.  See
  /// that method for details.
  void clear();

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

    iterator(const ThreadsafeList* parent, const Interval* interval);

    const ThreadsafeList* parent_ = nullptr;
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

  alignas(64) std::atomic<double> initial_time_ =
      std::numeric_limits<double>::signaling_NaN();
  // Pad memory to avoid false-sharing when accessing initial_time_
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  char unused_padding_initial_time_[64 - (sizeof(initial_time_) % 64)] = {};
  std::unique_ptr<Interval> interval_list_{};
  alignas(64) std::atomic<Interval*> most_recent_interval_{};
  // Pad memory to avoid false-sharing when accessing most_recent_interval_
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  char
      unused_padding_most_recent_interval_[64 - (sizeof(most_recent_interval_) %
                                                 64)] = {};
};

template <typename T>
bool operator==(const ThreadsafeList<T>& a, const ThreadsafeList<T>& b);

template <typename T>
bool operator!=(const ThreadsafeList<T>& a, const ThreadsafeList<T>& b);
}  // namespace domain::FunctionsOfTime::FunctionOfTimeHelpers
