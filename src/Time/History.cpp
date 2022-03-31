// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/History.hpp"

#include "Utilities/GenerateInstantiations.hpp"

namespace TimeSteppers {
template <typename T>
typename HistoryIterator<T>::reference HistoryIterator<T>::operator*() const {
  return history_->time_step_id_for_iterator(offset_).substep_time();
}

template <typename T>
typename HistoryIterator<T>::pointer HistoryIterator<T>::operator->() const {
  return &**this;
}

template <typename T>
typename HistoryIterator<T>::reference HistoryIterator<T>::operator[](
    const difference_type n) const {
  return history_->time_step_id_for_iterator(offset_ + n).substep_time();
}

template <typename T>
HistoryIterator<T>& HistoryIterator<T>::operator++() {
  ++offset_;
  return *this;
}

template <typename T>
HistoryIterator<T> HistoryIterator<T>::operator++(int) {
  return {history_, offset_++};
}

template <typename T>
HistoryIterator<T>& HistoryIterator<T>::operator--() {
  --offset_;
  return *this;
}

template <typename T>
HistoryIterator<T> HistoryIterator<T>::operator--(int) {
  return {history_, offset_--};
}

template <typename T>
HistoryIterator<T>& HistoryIterator<T>::operator+=(difference_type n) {
  offset_ += n;
  return *this;
}

template <typename T>
HistoryIterator<T>& HistoryIterator<T>::operator-=(difference_type n) {
  offset_ -= n;
  return *this;
}

template <typename T>
const TimeStepId& HistoryIterator<T>::time_step_id() const {
  return history_->time_step_id_for_iterator(offset_);
}

template <typename T>
MathWrapper<const T> HistoryIterator<T>::derivative() const {
  return history_->derivative_for_iterator(offset_);
}

template <typename T>
HistoryIterator<T>::HistoryIterator(const UntypedHistory<T>* const history,
                                    const difference_type offset)
    : history_(history), offset_(offset) {}

template <typename T>
HistoryIterator<T> operator+(HistoryIterator<T> it,
                             typename HistoryIterator<T>::difference_type n) {
  it += n;
  return it;
}

template <typename T>
HistoryIterator<T> operator+(typename HistoryIterator<T>::difference_type n,
                             HistoryIterator<T> it) {
  return std::move(it) + n;
}

template <typename T>
HistoryIterator<T> operator-(HistoryIterator<T> it,
                             typename HistoryIterator<T>::difference_type n) {
  return std::move(it) + (-n);
}

#define MATH_WRAPPER_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template class HistoryIterator<MATH_WRAPPER_TYPE(data)>;                   \
  template HistoryIterator<MATH_WRAPPER_TYPE(data)> operator+(               \
      HistoryIterator<MATH_WRAPPER_TYPE(data)> it,                           \
      typename HistoryIterator<MATH_WRAPPER_TYPE(data)>::difference_type n); \
  template HistoryIterator<MATH_WRAPPER_TYPE(data)> operator+(               \
      typename HistoryIterator<MATH_WRAPPER_TYPE(data)>::difference_type n,  \
      HistoryIterator<MATH_WRAPPER_TYPE(data)> it);                          \
  template HistoryIterator<MATH_WRAPPER_TYPE(data)> operator-(               \
      HistoryIterator<MATH_WRAPPER_TYPE(data)> it,                           \
      typename HistoryIterator<MATH_WRAPPER_TYPE(data)>::difference_type n);

GENERATE_INSTANTIATIONS(INSTANTIATE, (MATH_WRAPPER_TYPES))

#undef INSTANTIATE

#define FORWARD_HISTORY_ITERATOR_OP(T, op)                                     \
  bool operator op(const HistoryIterator<T>& a, const HistoryIterator<T>& b) { \
    return a.offset_ op b.offset_;                                             \
  }

// The friend functions are overloaded functions, not templates
#define OVERLOAD(_, data)                                            \
  typename HistoryIterator<MATH_WRAPPER_TYPE(data)>::difference_type \
  operator-(const HistoryIterator<MATH_WRAPPER_TYPE(data)>& a,       \
            const HistoryIterator<MATH_WRAPPER_TYPE(data)>& b) {     \
    return a.offset_ - b.offset_;                                    \
  }                                                                  \
  FORWARD_HISTORY_ITERATOR_OP(MATH_WRAPPER_TYPE(data), ==)           \
  FORWARD_HISTORY_ITERATOR_OP(MATH_WRAPPER_TYPE(data), !=)           \
  FORWARD_HISTORY_ITERATOR_OP(MATH_WRAPPER_TYPE(data), <)            \
  FORWARD_HISTORY_ITERATOR_OP(MATH_WRAPPER_TYPE(data), >)            \
  FORWARD_HISTORY_ITERATOR_OP(MATH_WRAPPER_TYPE(data), <=)           \
  FORWARD_HISTORY_ITERATOR_OP(MATH_WRAPPER_TYPE(data), >=)

GENERATE_INSTANTIATIONS(OVERLOAD, (MATH_WRAPPER_TYPES))

#undef OVERLOAD
#undef FORWARD_HISTORY_ITERATOR_OP
#undef MATH_WRAPPER_TYPE
}  // namespace TimeSteppers
