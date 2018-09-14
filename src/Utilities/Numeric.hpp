// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include "Utilities/Gsl.hpp"

/// C++ STL code present in C++2b.
namespace cpp2b {
/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::iota that is constexpr;
 * taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/numeric
 */
template <class ForwardIterator, class T>
constexpr void iota(ForwardIterator first, ForwardIterator last, T value) {
  for (; first != last; ++first, (void)++value) {
    *first = value;
  }
}

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::accumulate that is constexpr;
 * taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/numeric
 */
template <class InputIt, class T>
constexpr T accumulate(InputIt first, InputIt last, T init) {
  for (; first != last; ++first) {
    init = std::move(init) + *first;  // std::move since C++20
  }
  return init;
}
}  // namespace cpp2b
