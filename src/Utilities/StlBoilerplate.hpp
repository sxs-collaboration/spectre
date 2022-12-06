// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>

/// Classes to generate repetitive code needed for STL compatibility.
namespace stl_boilerplate {
/// Base class to generate methods for a random-access iterator.  This
/// class takes the derived class and the (optionally `const`)
/// `value_type` of the iterator.  The exposed `value_type` alias will
/// always be non-const, as required by the standard, but the template
/// parameter's constness will affect the `reference` and `pointer`
/// aliases.
///
/// The derived class must implement `operator*`, `operator+=`,
/// `operator-` (of two iterators), and `operator==`.
///
/// \snippet Test_StlBoilerplate.cpp RandomAccessIterator
///
/// This class is inspired by Boost's `iterator_facade`, except that
/// that has a lot of special cases, some of which work poorly because
/// they predate C++11.
template <typename Iter, typename ValueType>
class RandomAccessIterator {
 protected:
  RandomAccessIterator() = default;
  RandomAccessIterator(const RandomAccessIterator&) = default;
  RandomAccessIterator(RandomAccessIterator&&) = default;
  RandomAccessIterator& operator=(const RandomAccessIterator&) = default;
  RandomAccessIterator& operator=(RandomAccessIterator&&) = default;
  ~RandomAccessIterator() = default;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = std::remove_const_t<ValueType>;
  using reference = ValueType&;
  using pointer = ValueType*;
  using difference_type = std::ptrdiff_t;

  pointer operator->() const { return &**dthis(); }

  Iter& operator++() { return *dthis() += 1; }
  Iter operator++(int) {
    const auto ret = *dthis();
    ++*dthis();
    return ret;
  }
  Iter& operator--() { return *dthis() += -1; }
  Iter operator--(int) {
    const auto ret = *dthis();
    --*dthis();
    return ret;
  }

  Iter& operator-=(const difference_type n) { return *dthis() += -n; }

  reference operator[](const difference_type n) const {
    auto temp = *dthis();
    temp += n;
    return *temp;
  }

 private:
  // derived this
  Iter* dthis() { return static_cast<Iter*>(this); }
  const Iter* dthis() const { return static_cast<const Iter*>(this); }
};

template <typename Iter, typename ValueType>
bool operator!=(const RandomAccessIterator<Iter, ValueType>& a,
                const RandomAccessIterator<Iter, ValueType>& b) {
  return not(static_cast<const Iter&>(a) == static_cast<const Iter&>(b));
}

template <typename Iter, typename ValueType>
bool operator<(const RandomAccessIterator<Iter, ValueType>& a,
               const RandomAccessIterator<Iter, ValueType>& b) {
  return static_cast<const Iter&>(b) - static_cast<const Iter&>(a) > 0;
}

template <typename Iter, typename ValueType>
bool operator>(const RandomAccessIterator<Iter, ValueType>& a,
               const RandomAccessIterator<Iter, ValueType>& b) {
  return b < a;
}

template <typename Iter, typename ValueType>
bool operator<=(const RandomAccessIterator<Iter, ValueType>& a,
                const RandomAccessIterator<Iter, ValueType>& b) {
  return not(a > b);
}

template <typename Iter, typename ValueType>
bool operator>=(const RandomAccessIterator<Iter, ValueType>& a,
                const RandomAccessIterator<Iter, ValueType>& b) {
  return not(a < b);
}

template <typename Iter, typename ValueType>
Iter operator+(
    const RandomAccessIterator<Iter, ValueType>& a,
    const typename RandomAccessIterator<Iter, ValueType>::difference_type& n) {
  auto result = static_cast<const Iter&>(a);
  result += n;
  return result;
}

template <typename Iter, typename ValueType>
Iter operator+(
    const typename RandomAccessIterator<Iter, ValueType>::difference_type& n,
    const RandomAccessIterator<Iter, ValueType>& a) {
  return a + n;
}

template <typename Iter, typename ValueType>
Iter operator-(
    const RandomAccessIterator<Iter, ValueType>& a,
    const typename RandomAccessIterator<Iter, ValueType>::difference_type& n) {
  auto result = static_cast<const Iter&>(a);
  result -= n;
  return result;
}
}  // namespace stl_boilerplate
