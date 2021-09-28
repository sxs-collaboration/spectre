// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <iterator>
#include <tuple>
#include <utility>

#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"

// IWYU pragma: no_include <unordered_set>  // for swap?

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

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
  History(const History&) = default;
  History(History&&) = default;
  History& operator=(const History&) = default;
  History& operator=(History&&) = default;
  ~History() = default;

  explicit History(const size_t integration_order)
      : integration_order_(integration_order) {}

  /// Add a new set of values to the end of the history.
  void insert(const TimeStepId& time_step_id, const DerivVars& deriv);

  /// Add a new derivative to the front of the history.  This is often
  /// convenient for setting initial data.  The function value must be
  /// set separately before this object is used, using `most_recent_value()`.
  void insert_initial(TimeStepId time_step_id, DerivVars deriv);

  /// The most recent value of the integrated variables.
  /// @{
  Vars& most_recent_value() { return most_recent_value_; }
  const Vars& most_recent_value() const { return most_recent_value_; }
  /// @}

  /// Mark all data before the passed point in history as unneeded so
  /// it can be removed.  Calling this directly should not often be
  /// necessary, as it is handled internally by the time steppers.
  void mark_unneeded(const const_iterator& first_needed);

  /// These iterators directly return the Time of the past values.
  /// The derivative data can be accessed through the iterators using
  /// HistoryIterator::derivative().
  /// @{
  const_iterator begin() const {
    return data_.begin() +
           static_cast<typename decltype(data_.begin())::difference_type>(
               first_needed_entry_);
  }
  const_iterator end() const { return data_.end(); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  /// @}

  size_type size() const { return capacity() - first_needed_entry_; }
  size_type capacity() const { return data_.size(); }
  void shrink_to_fit();

  /// These return the past times.  The other data can be accessed
  /// through HistoryIterator methods.
  /// @{
  const_reference operator[](size_type n) const {
    return *(begin() + static_cast<difference_type>(n));
  }

  const_reference front() const { return *begin(); }
  const_reference back() const {
    return std::get<0>(data_.back()).substep_time();
  }
  /// @}

  /// Get or set the current order of integration.  TimeSteppers may
  /// impose restrictions on the valid values.
  /// @{
  size_t integration_order() const { return integration_order_; }
  void integration_order(const size_t integration_order) {
    integration_order_ = integration_order;
  }
  /// @}

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) {  // NOLINT
    // Don't send cached allocations.  This object is probably going
    // to be thrown away after serialization, so we take the easy
    // route of just throwing them away.
    shrink_to_fit();
    p | most_recent_value_;
    p | data_;
    p | integration_order_;
  }

 private:
  Vars most_recent_value_{};
  std::deque<std::tuple<TimeStepId, DerivVars>> data_;
  size_t first_needed_entry_{0};
  size_t integration_order_{0};
};

/// \ingroup TimeSteppersGroup
/// Iterator over history data used by a TimeStepper.  See History for
/// details.
template <typename Vars, typename DerivVars>
class HistoryIterator {
  using Base =
      typename std::deque<std::tuple<TimeStepId, DerivVars>>::const_iterator;

 public:
  using iterator_category =
      typename std::iterator_traits<Base>::iterator_category;
  using value_type = Time;
  using difference_type = typename std::iterator_traits<Base>::difference_type;
  using pointer = const value_type*;
  using reference = const value_type&;

  HistoryIterator() = default;

  reference operator*() const { return std::get<0>(*base_).substep_time(); }
  pointer operator->() const { return &**this; }
  reference operator[](const difference_type n) const {
    return std::get<0>(base_[n]).substep_time();
  }
  HistoryIterator& operator++() {
    ++base_;
    return *this;
  }
  // clang-tidy: return const... Really? What?
  HistoryIterator operator++(int) { return base_++; }  // NOLINT
  HistoryIterator& operator--() {
    --base_;
    return *this;
  }
  // clang-tidy: return const... Really? What?
  HistoryIterator operator--(int) { return base_--; }  // NOLINT
  HistoryIterator& operator+=(difference_type n) {
    base_ += n;
    return *this;
  }
  HistoryIterator& operator-=(difference_type n) {
    base_ -= n;
    return *this;
  }

  const TimeStepId& time_step_id() const { return std::get<0>(*base_); }
  const DerivVars& derivative() const { return std::get<1>(*base_); }

 private:
  friend class History<Vars, DerivVars>;

  friend difference_type operator-(const HistoryIterator& a,
                                   const HistoryIterator& b) {
    return a.base_ - b.base_;
  }

#define FORWARD_HISTORY_ITERATOR_OP(op)               \
  friend bool operator op(const HistoryIterator& a,   \
                          const HistoryIterator& b) { \
    return a.base_ op b.base_;                        \
  }
  FORWARD_HISTORY_ITERATOR_OP(==)
  FORWARD_HISTORY_ITERATOR_OP(!=)
  FORWARD_HISTORY_ITERATOR_OP(<)
  FORWARD_HISTORY_ITERATOR_OP(>)
  FORWARD_HISTORY_ITERATOR_OP(<=)
  FORWARD_HISTORY_ITERATOR_OP(>=)
#undef FORWARD_HISTORY_ITERATOR_OP

  // clang-tidy: google-explicit-constructor - private
  HistoryIterator(Base base) : base_(std::move(base)) {}  // NOLINT

  Base base_{};
};

// ================================================================

template <typename Vars, typename DerivVars>
void History<Vars, DerivVars>::insert(const TimeStepId& time_step_id,
                                      const DerivVars& deriv) {
  if (first_needed_entry_ == 0) {
    data_.emplace_back(time_step_id, deriv);
  } else {
    // Reuse resources from an old entry.
    auto& old_entry = data_.front();
    // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
    std::get<0>(old_entry) = time_step_id;
    std::get<1>(old_entry) = deriv;
    data_.push_back(std::move(old_entry));
    data_.pop_front();
    --first_needed_entry_;
  }
}

template <typename Vars, typename DerivVars>
inline void History<Vars, DerivVars>::insert_initial(TimeStepId time_step_id,
                                                     DerivVars deriv) {
  // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
  data_.emplace_front(std::move(time_step_id), std::move(deriv));
}

template <typename Vars, typename DerivVars>
inline void History<Vars, DerivVars>::mark_unneeded(
    const const_iterator& first_needed) {
  first_needed_entry_ = static_cast<size_t>(first_needed.base_ - data_.begin());
}

template <typename Vars, typename DerivVars>
inline void History<Vars, DerivVars>::shrink_to_fit() {
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
    typename HistoryIterator<Vars, DerivVars>::difference_type n) {
  it += n;
  return it;
}

template <typename Vars, typename DerivVars>
inline HistoryIterator<Vars, DerivVars> operator+(
    typename HistoryIterator<Vars, DerivVars>::difference_type n,
    HistoryIterator<Vars, DerivVars> it) {
  return std::move(it) + n;
}

template <typename Vars, typename DerivVars>
inline HistoryIterator<Vars, DerivVars> operator-(
    HistoryIterator<Vars, DerivVars> it,
    typename HistoryIterator<Vars, DerivVars>::difference_type n) {
  return std::move(it) + (-n);
}
}  // namespace TimeSteppers
