// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <iterator>
#include <pup.h>
#include <tuple>
#include <utility>

#include "Parallel/PupStlCpp11.hpp"
#include "Time/Time.hpp"

namespace TimeSteppers {

template <typename Vars, typename DerivVars>
class HistoryIterator;

/// \ingroup TimeSteppersGroup
/// History data used by a TimeStepper.
/// \tparam Vars type of variables being integrated
/// \tparam DerivVars type of derivative variables
template <typename Vars, typename DerivVars>
class History {
 public:
  using value_type = Time;
  using const_reference = const value_type&;
  using const_iterator = HistoryIterator<Vars, DerivVars>;
  using difference_type =
      typename std::iterator_traits<const_iterator>::difference_type;
  using size_type = size_t;

  History() = default;
  History(const History&) = delete;
  History(History&&) = default;
  History& operator=(const History&) = delete;
  History& operator=(History&&) = default;
  ~History() = default;

  /// Add a new set of values to the end of the history.  `deriv` is
  /// either left unchanged or set to a previously inserted `deriv`
  /// value.  The different argument types for `value` and `deriv`
  /// reflect the common use of this class, which wants `value` to
  /// remain unchanged but does not care about `deriv`.
  void insert(Time time, const Vars& value, DerivVars&& deriv) noexcept;

  /// Add a new set of values to the front of the history.  This is
  /// often convenient for setting initial data.
  void insert_initial(Time time, Vars value, DerivVars deriv) noexcept;

  /// Mark all data before the passed point in history as unneeded so
  /// it can be removed.  Calling this directly should not often be
  /// necessary, as it is handled internally by the time steppers.
  void mark_unneeded(const const_iterator& first_needed) noexcept;

  /// These iterators directly return the Time of the past values.
  /// The other data can be accessed through the iterators using
  /// HistoryIterator::value() and HistoryIterator::derivative().
  //@{
  const_iterator begin() const noexcept {
    return data_.begin() +
           static_cast<typename decltype(data_.begin())::difference_type>(
               first_needed_entry_);
  }
  const_iterator end() const noexcept { return data_.end(); }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }
  //@}

  size_type size() const noexcept { return capacity() - first_needed_entry_; }
  size_type capacity() const noexcept { return data_.size(); }
  void shrink_to_fit() noexcept;

  /// These return the past times.  The other data can be accessed
  /// through HistoryIterator methods.
  //@{
  const_reference operator[](size_type n) const noexcept {
    return *(begin() + static_cast<difference_type>(n));
  }

  const_reference front() const noexcept { return *begin(); }
  const_reference back() const noexcept { return std::get<0>(data_.back()); }
  //@}

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    // Don't send cached allocations.  This object is probably going
    // to be thrown away after serialization, so we take the easy
    // route of just throwing them away.
    shrink_to_fit();
    p | data_;
  }

 private:
  std::deque<std::tuple<Time, Vars, DerivVars>> data_;
  size_t first_needed_entry_{0};
};

/// \ingroup TimeSteppersGroup
/// Iterator over history data used by a TimeStepper.  See History for
/// details.
template <typename Vars, typename DerivVars>
class HistoryIterator {
  using Base =
      typename std::deque<std::tuple<Time, Vars, DerivVars>>::const_iterator;

 public:
  using iterator_category =
      typename std::iterator_traits<Base>::iterator_category;
  using value_type = Time;
  using difference_type = typename std::iterator_traits<Base>::difference_type;
  using pointer = const value_type*;
  using reference = const value_type&;

  HistoryIterator() = default;

  reference operator*() const noexcept { return std::get<0>(*base_); }
  pointer operator->() const noexcept { return &std::get<0>(*base_); }
  reference operator[](difference_type n) const noexcept {
    return std::get<0>(base_[n]);
  }
  HistoryIterator& operator++() noexcept { ++base_; return *this; }
  // clang-tidy: return const... Really? What?
  HistoryIterator operator++(int) noexcept { return base_++; }  // NOLINT
  HistoryIterator& operator--() noexcept { --base_; return *this; }
  // clang-tidy: return const... Really? What?
  HistoryIterator operator--(int) noexcept { return base_--; }  // NOLINT
  HistoryIterator& operator+=(difference_type n) noexcept {
    base_ += n;
    return *this;
  }
  HistoryIterator& operator-=(difference_type n) noexcept {
    base_ -= n;
    return *this;
  }

  const Vars& value() const noexcept { return std::get<1>(*base_); }
  const DerivVars& derivative() const noexcept { return std::get<2>(*base_); }

 private:
  friend class History<Vars, DerivVars>;

  friend difference_type operator-(const HistoryIterator& a,
                                   const HistoryIterator& b) noexcept {
    return a.base_ - b.base_;
  }

#define FORWARD_HISTORY_ITERATOR_OP(op)                        \
  friend bool operator op(const HistoryIterator& a,            \
                          const HistoryIterator& b) noexcept { \
    return a.base_ op b.base_;                                 \
  }
  FORWARD_HISTORY_ITERATOR_OP(==)
  FORWARD_HISTORY_ITERATOR_OP(!=)
  FORWARD_HISTORY_ITERATOR_OP(<)
  FORWARD_HISTORY_ITERATOR_OP(>)
  FORWARD_HISTORY_ITERATOR_OP(<=)
  FORWARD_HISTORY_ITERATOR_OP(>=)
#undef FORWARD_HISTORY_ITERATOR_OP

  // clang-tidy: google-explicit-constructor - private
  HistoryIterator(Base base) noexcept : base_(std::move(base)) {}  // NOLINT

  Base base_{};
};

// ================================================================

template <typename Vars, typename DerivVars>
void History<Vars, DerivVars>::insert(Time time, const Vars& value,
                                      DerivVars&& deriv) noexcept {
  if (first_needed_entry_ == 0) {
    // clang-tidy: move of trivially-copyable type
    data_.emplace_back(std::move(time), value, deriv);  // NOLINT
  } else {
    // Move an unneeded entry into the arguments so the caller can
    // reuse any resources the entry contained.
    using std::swap;
    auto& old_entry = data_.front();
    // clang-tidy: move of trivially-copyable type
    std::get<0>(old_entry) = std::move(time);  // NOLINT
    std::get<1>(old_entry) = value;
    swap(std::get<2>(old_entry), deriv);
    data_.push_back(std::move(old_entry));
    data_.pop_front();
    --first_needed_entry_;
  }
}

template <typename Vars, typename DerivVars>
inline void History<Vars, DerivVars>::insert_initial(Time time, Vars value,
                                                     DerivVars deriv) noexcept {
  // clang-tidy: move of trivially-copyable type
  data_.emplace_front(std::move(time), // NOLINT
                      std::move(value), std::move(deriv));
}

template <typename Vars, typename DerivVars>
inline void History<Vars, DerivVars>::mark_unneeded(
    const const_iterator& first_needed) noexcept {
  first_needed_entry_ = static_cast<size_t>(first_needed.base_ - data_.begin());
}

template <typename Vars, typename DerivVars>
inline void History<Vars, DerivVars>::shrink_to_fit() noexcept {
  data_.erase(
      data_.begin(),
      data_.begin() +
          static_cast<typename decltype(data_.begin())::difference_type>(
              first_needed_entry_));
  first_needed_entry_ = 0;
}

template <typename Vars, typename DerivVars>
inline HistoryIterator<Vars, DerivVars> operator+(
    HistoryIterator<Vars, DerivVars> it,
    typename HistoryIterator<Vars, DerivVars>::difference_type n) noexcept {
  it += n;
  return it;
}

template <typename Vars, typename DerivVars>
inline HistoryIterator<Vars, DerivVars> operator+(
    typename HistoryIterator<Vars, DerivVars>::difference_type n,
    HistoryIterator<Vars, DerivVars> it) noexcept {
  return std::move(it) + n;
}

template <typename Vars, typename DerivVars>
inline HistoryIterator<Vars, DerivVars> operator-(
    HistoryIterator<Vars, DerivVars> it,
    typename HistoryIterator<Vars, DerivVars>::difference_type n) noexcept {
  return std::move(it) + (-n);
}
}  // namespace TimeSteppers
