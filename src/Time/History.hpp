// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <iterator>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"

// IWYU pragma: no_include <unordered_set>  // for swap?

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace TimeSteppers {

/// \cond
template <typename Vars>
class DerivIterator;
template <typename T>
class HistoryIterator;
/// \endcond

#if defined(__GNUC__) && not defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif  // defined(__GNUC__) && not defined(__clang__)
/// \ingroup TimeSteppersGroup
/// Type-erased history data used by a TimeStepper.
/// \tparam T One of the types in \ref MATH_WRAPPER_TYPES
template <typename T>
class UntypedHistory {
 protected:
  UntypedHistory() = default;
  UntypedHistory(const UntypedHistory&) = default;
  UntypedHistory(UntypedHistory&&) = default;
  UntypedHistory& operator=(const UntypedHistory&) = default;
  UntypedHistory& operator=(UntypedHistory&&) = default;
  ~UntypedHistory() = default;

 public:
  using value_type = Time;
  using const_reference = const value_type&;
  using const_iterator = HistoryIterator<T>;
  using difference_type =
      typename std::iterator_traits<const_iterator>::difference_type;
  using size_type = size_t;

  /// Mark all data before the passed point in history as unneeded so
  /// it can be removed.  Calling this directly should not often be
  /// necessary, as it is handled internally by the time steppers.
  virtual void mark_unneeded(const const_iterator& first_needed) = 0;

  /// These iterators directly return the Time of the past values.
  /// The derivative data can be accessed through the iterators using
  /// HistoryIterator::derivative().
  /// @{
  virtual const_iterator begin() const = 0;
  virtual const_iterator end() const = 0;
  virtual const_iterator cbegin() const = 0;
  virtual const_iterator cend() const = 0;
  /// @}

  virtual size_type size() const = 0;
  virtual size_type capacity() const = 0;

  /// These return the past times.  The other data can be accessed
  /// through HistoryIterator methods.
  /// @{
  virtual const_reference operator[](size_type n) const = 0;

  virtual const_reference front() const = 0;
  virtual const_reference back() const = 0;
  /// @}

  /// Get the current order of integration.
  virtual size_t integration_order() const = 0;

 private:
  friend class HistoryIterator<T>;
  virtual const TimeStepId& time_step_id_for_iterator(
      const difference_type offset) const = 0;
  virtual MathWrapper<const T> derivative_for_iterator(
      const difference_type offset) const = 0;
};

/// \ingroup TimeSteppersGroup
/// History data used by a TimeStepper.
/// \tparam DerivVars type of the derivatives
template <typename DerivVars>
class History final : public UntypedHistory<math_wrapper_type<DerivVars>> {
  using Base = UntypedHistory<math_wrapper_type<DerivVars>>;

 public:
  using value_type = typename Base::value_type;
  using const_reference = typename Base::const_reference;
  using const_iterator = typename Base::const_iterator;
  using difference_type = typename Base::difference_type;
  using size_type = typename Base::size_type;

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
  /// convenient for setting initial data.
  void insert_initial(TimeStepId time_step_id, DerivVars deriv);

  void mark_unneeded(const const_iterator& first_needed) override;

  const_iterator begin() const override {
    return {this, static_cast<difference_type>(first_needed_entry_)};
  }
  const_iterator end() const override {
    return {this, static_cast<difference_type>(data_.size())};
  }
  const_iterator cbegin() const override { return begin(); }
  const_iterator cend() const override { return end(); }

  size_type size() const override { return capacity() - first_needed_entry_; }
  size_type capacity() const override { return data_.size(); }
  void shrink_to_fit();

  /// These return the past times.  The other data can be accessed
  /// through HistoryIterator methods.
  /// @{
  const_reference operator[](size_type n) const override {
    return *(begin() + static_cast<difference_type>(n));
  }

  const_reference front() const override { return *begin(); }
  const_reference back() const override {
    return std::get<0>(data_.back()).substep_time();
  }
  /// @}

  /// Iterate over the typed derivatives.  See the standard iterators
  /// for type-erased use.
  /// @{
  DerivIterator<DerivVars> derivatives_begin() const {
    return DerivIterator<DerivVars>(
        data_.begin() + static_cast<difference_type>(first_needed_entry_));
  }
  DerivIterator<DerivVars> derivatives_end() const {
    return DerivIterator<DerivVars>(data_.end());
  }
  /// @}

  size_t integration_order() const override { return integration_order_; }

  /// Set the current order of integration.  TimeSteppers may impose
  /// restrictions on the valid values.
  void integration_order(const size_t integration_order) {
    integration_order_ = integration_order;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    // Don't send cached allocations.  This object is probably going
    // to be thrown away after serialization, so we take the easy
    // route of just throwing them away.
    shrink_to_fit();
    p | data_;
    p | integration_order_;
  }

 private:
  const TimeStepId& time_step_id_for_iterator(
      const difference_type offset) const override {
    return std::get<0>(*(data_.begin() + offset));
  }

  MathWrapper<math_wrapper_type<const DerivVars>> derivative_for_iterator(
      const difference_type offset) const override {
    return make_math_wrapper(std::get<1>(*(data_.begin() + offset)));
  }

  std::deque<std::tuple<TimeStepId, DerivVars>> data_;
  size_t first_needed_entry_{0};
  size_t integration_order_{0};
};
#if defined(__GNUC__) && not defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && not defined(__clang__)

/// \ingroup TimeSteppersGroup
/// Iterator over history data used by a TimeStepper.  See History for
/// details.
template <typename T>
class HistoryIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = Time;
  using difference_type = std::ptrdiff_t;
  using pointer = const value_type*;
  using reference = const value_type&;

  HistoryIterator() = default;

  reference operator*() const;
  pointer operator->() const;
  reference operator[](const difference_type n) const;
  HistoryIterator& operator++();
  // clang-tidy: return const... Really? What?
  HistoryIterator operator++(int);  // NOLINT
  HistoryIterator& operator--();
  // clang-tidy: return const... Really? What?
  HistoryIterator operator--(int);  // NOLINT
  HistoryIterator& operator+=(difference_type n);
  HistoryIterator& operator-=(difference_type n);

  const TimeStepId& time_step_id() const;
  MathWrapper<const T> derivative() const;

 private:
  template <typename DerivVars>
  friend class History;

  friend difference_type operator-(const HistoryIterator& a,
                                   const HistoryIterator& b);

  friend bool operator==(const HistoryIterator& a, const HistoryIterator& b);
  friend bool operator!=(const HistoryIterator& a, const HistoryIterator& b);
  friend bool operator<(const HistoryIterator& a, const HistoryIterator& b);
  friend bool operator>(const HistoryIterator& a, const HistoryIterator& b);
  friend bool operator<=(const HistoryIterator& a, const HistoryIterator& b);
  friend bool operator>=(const HistoryIterator& a, const HistoryIterator& b);

  HistoryIterator(const UntypedHistory<T>* const history,
                  const difference_type offset);

  const UntypedHistory<T>* history_;
  difference_type offset_;
};

/// \ingroup TimeSteppersGroup
/// Iterator over the derivatives stored in a History object.
///
/// This returns typed data, and so cannot be obtained from an
/// UntypedHistory.  Type-erased derivatives can be obtained from
/// HistoryIterator::derivative().
template <typename DerivVars>
class DerivIterator {
  using Base =
      typename std::deque<std::tuple<TimeStepId, DerivVars>>::const_iterator;

 public:
  using iterator_category = typename Base::iterator_category;
  using value_type = DerivVars;
  using difference_type = typename Base::difference_type;
  using pointer = const value_type*;
  using reference = const value_type&;

  DerivIterator() = default;

  reference operator*() const { return std::get<1>(*base_); }
  pointer operator->() const { return &**this; }
  reference operator[](const difference_type n) const {
    return std::get<1>(base_[n]);
  }
  DerivIterator& operator++() {
    ++base_;
    return *this;
  }
  DerivIterator operator++(int) { return DerivIterator(base_++); }
  DerivIterator& operator--() {
    --base_;
    return *this;
  }
  DerivIterator operator--(int) { return DerivIterator(base_--); }
  DerivIterator& operator+=(difference_type n) {
    base_ += n;
    return *this;
  }
  DerivIterator& operator-=(difference_type n) {
    base_ -= n;
    return *this;
  }

  const TimeStepId& time_step_id() const { return std::get<0>(*base_); }

 private:
  friend class History<DerivVars>;

  friend difference_type operator-(const DerivIterator& a,
                                   const DerivIterator& b) {
    return a.base_ - b.base_;
  }

#define FORWARD_DERIV_ITERATOR_OP(op)                                       \
  friend bool operator op(const DerivIterator& a, const DerivIterator& b) { \
    return a.base_ op b.base_;                                              \
  }
  FORWARD_DERIV_ITERATOR_OP(==)
  FORWARD_DERIV_ITERATOR_OP(!=)
  FORWARD_DERIV_ITERATOR_OP(<)
  FORWARD_DERIV_ITERATOR_OP(>)
  FORWARD_DERIV_ITERATOR_OP(<=)
  FORWARD_DERIV_ITERATOR_OP(>=)
#undef FORWARD_DERIV_ITERATOR_OP

  explicit DerivIterator(Base base) : base_(std::move(base)) {}

  Base base_;
};

// ================================================================

template <typename DerivVars>
void History<DerivVars>::insert(const TimeStepId& time_step_id,
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

template <typename DerivVars>
inline void History<DerivVars>::insert_initial(TimeStepId time_step_id,
                                               DerivVars deriv) {
  // NOLINTNEXTLINE(hicpp-move-const-arg,performance-move-const-arg)
  data_.emplace_front(std::move(time_step_id), std::move(deriv));
}

template <typename DerivVars>
inline void History<DerivVars>::mark_unneeded(
    const const_iterator& first_needed) {
  first_needed_entry_ = static_cast<size_t>(first_needed.offset_);
}

template <typename DerivVars>
inline void History<DerivVars>::shrink_to_fit() {
  data_.erase(
      data_.begin(),
      data_.begin() +
          static_cast<typename decltype(data_.begin())::difference_type>(
              first_needed_entry_));
  first_needed_entry_ = 0;
}

template <typename T>
HistoryIterator<T> operator+(HistoryIterator<T> it,
                             typename HistoryIterator<T>::difference_type n);

template <typename T>
HistoryIterator<T> operator+(typename HistoryIterator<T>::difference_type n,
                             HistoryIterator<T> it);

template <typename T>
HistoryIterator<T> operator-(HistoryIterator<T> it,
                             typename HistoryIterator<T>::difference_type n);

template <typename DerivVars>
DerivIterator<DerivVars> operator+(
    DerivIterator<DerivVars> it,
    const typename DerivIterator<DerivVars>::difference_type n) {
  it += n;
  return it;
}

template <typename DerivVars>
DerivIterator<DerivVars> operator+(
    const typename DerivIterator<DerivVars>::difference_type n,
    DerivIterator<DerivVars> it) {
  return std::move(it) + n;
}

template <typename DerivVars>
DerivIterator<DerivVars> operator-(
    DerivIterator<DerivVars> it,
    const typename DerivIterator<DerivVars>::difference_type n) {
  return std::move(it) + (-n);
}
}  // namespace TimeSteppers
