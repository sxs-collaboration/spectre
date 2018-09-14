// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <functional>
#include <iterator>

#include "Utilities/Gsl.hpp"

namespace cpp20 {
/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::swap that is constexpr;
 * taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/type_traits
 */
template <class T>
constexpr void swap(T& a, T& b) noexcept {
  T c(std::move(a));
  a = std::move(b);
  b = std::move(c);
}

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::iter_swap that is constexpr;
 * taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/type_traits
 */
template <class ForwardIt1, class ForwardIt2>
constexpr void iter_swap(ForwardIt1 a, ForwardIt2 b) {
  swap(*a, *b);
}

namespace detail {
/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::reverse that is constexpr, for bidirectional
 * iterators; taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/algorithm
 */
template <class BidirIt>
constexpr void reverse(BidirIt first, BidirIt last,
                       std::bidirectional_iterator_tag /* unused */) {
  while (first != last) {
    if (first == --last) {
      break;
    }
    cpp20::iter_swap(first, last);
    ++first;
  }
}

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::reverse that is constexpr, for random access
 * iterators; taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/algorithm
 */
template <class RandomAccessIterator>
constexpr void reverse(RandomAccessIterator first, RandomAccessIterator last,
                       std::random_access_iterator_tag /* unused */) {
  if (first != last) {
    for (; first < --last; ++first) {
      cpp20::iter_swap(first, last);
    }
  }
}
}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::reverse that is constexpr;
 * taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/algorithm
 */
template <class BidirectionalIterator>
constexpr void reverse(BidirectionalIterator first,
                       BidirectionalIterator last) {
  cpp20::detail::reverse(first, last,
                         typename std::iterator_traits<
                             BidirectionalIterator>::iterator_category());
}

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::next_permutation that is constexpr,
 * for a generic comparator; taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/algorithm
 */
template <class Compare, class BidirectionalIterator>
constexpr bool next_permutation(BidirectionalIterator first,
                                BidirectionalIterator last, Compare comp) {
  BidirectionalIterator i = last;
  if (first == last || first == --i) {
    return false;
  }
  while (true) {
    BidirectionalIterator ip1 = i;
    if (comp(*--i, *ip1)) {
      BidirectionalIterator j = last;
      while (!comp(*i, *--j)) {
      }
      cpp20::swap(*i, *j);
      cpp20::reverse(ip1, last);
      return true;
    }
    if (i == first) {
      cpp20::reverse(first, last);
      return false;
    }
  }
}

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::next_permutation that is constexpr,
 * with less as the comparator; taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/algorithm
 */
template <class BidirectionalIterator>
constexpr bool next_permutation(BidirectionalIterator first,
                                BidirectionalIterator last) {
  return cpp20::next_permutation(
      first, last,
      std::less<
          typename std::iterator_traits<BidirectionalIterator>::value_type>());
}
}  // namespace cpp20
