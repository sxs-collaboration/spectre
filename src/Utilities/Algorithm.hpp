// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

namespace cpp20 {
/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::swap that is constexpr;
 * taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/type_traits
 */
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class T>
KOKKOS_FUNCTION constexpr void swap(T& a, T& b) noexcept(
    std::is_nothrow_move_constructible_v<T> and
    std::is_nothrow_move_assignable_v<T>) {
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
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class ForwardIt1, class ForwardIt2>
KOKKOS_FUNCTION constexpr void iter_swap(ForwardIt1 a, ForwardIt2 b) {
  cpp20::swap(*a, *b);
}

namespace detail {
/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::reverse that is constexpr, for bidirectional
 * iterators; taken from the LLVM source at
 * https://github.com/llvm-mirror/libcxx/blob/master/include/algorithm
 */
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class BidirIt>
KOKKOS_FUNCTION constexpr void reverse(
    BidirIt first, BidirIt last, std::bidirectional_iterator_tag /* unused */) {
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
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class RandomAccessIterator>
KOKKOS_FUNCTION constexpr void reverse(
    RandomAccessIterator first, RandomAccessIterator last,
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
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class BidirectionalIterator>
KOKKOS_FUNCTION constexpr void reverse(BidirectionalIterator first,
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
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Compare, class BidirectionalIterator>
KOKKOS_FUNCTION constexpr bool next_permutation(BidirectionalIterator first,
                                                BidirectionalIterator last,
                                                Compare comp) {
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
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class BidirectionalIterator>
KOKKOS_FUNCTION constexpr bool next_permutation(BidirectionalIterator first,
                                                BidirectionalIterator last) {
  return cpp20::next_permutation(
      first, last,
      std::less<
          typename std::iterator_traits<BidirectionalIterator>::value_type>());
}

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::find that is constexpr
 */
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class InputIt, class T>
  requires std::contiguous_iterator<InputIt>
KOKKOS_FUNCTION constexpr InputIt find(InputIt first, InputIt last,
                                       const T& value) {
  for (; first != last; ++first) {
    if (*first == value) {
      return first;
    }
  }
  return last;
}

// Non-contiguous iterators use host-only overloads with the same behavior, see
// `Algorithm_detail::device_iterable` below.
/// \cond
template <class InputIt, class T>
constexpr InputIt find(InputIt first, InputIt last, const T& value) {
  for (; first != last; ++first) {
    if (*first == value) {
      return first;
    }
  }
  return last;
}
/// \endcond

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::find_if that is constexpr
 */
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class InputIt, class UnaryPredicate>
  requires std::contiguous_iterator<InputIt>
KOKKOS_FUNCTION constexpr InputIt find_if(InputIt first, InputIt last,
                                          UnaryPredicate p) {
  for (; first != last; ++first) {
    if (p(*first)) {
      return first;
    }
  }
  return last;
}

/// \cond
template <class InputIt, class UnaryPredicate>
constexpr InputIt find_if(InputIt first, InputIt last, UnaryPredicate p) {
  for (; first != last; ++first) {
    if (p(*first)) {
      return first;
    }
  }
  return last;
}
/// \endcond

/*!
 * \ingroup UtilitiesGroup
 * Reimplementation of std::find_if_not that is constexpr
 */
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class InputIt, class UnaryPredicate>
  requires std::contiguous_iterator<InputIt>
KOKKOS_FUNCTION constexpr InputIt find_if_not(InputIt first, InputIt last,
                                              UnaryPredicate q) {
  for (; first != last; ++first) {
    if (!q(*first)) {
      return first;
    }
  }
  return last;
}

/// \cond
template <class InputIt, class UnaryPredicate>
constexpr InputIt find_if_not(InputIt first, InputIt last, UnaryPredicate q) {
  for (; first != last; ++first) {
    if (!q(*first)) {
      return first;
    }
  }
  return last;
}
/// \endcond
}  // namespace cpp20

namespace Algorithm_detail {
using std::begin;
// The containers that the device-callable `alg` functions loop over. Other
// containers, such as the standard library's node-based containers, use
// host-only overloads. For them, the loops would warn in CUDA builds about
// calling host functions from device functions, e.g. because their iterators
// have implicitly constexpr comparison operators.
//
// Like the standard library algorithms, the loops call a copy of the function
// objects passed to them, so the caller's function objects aren't modified.
template <typename Container>
concept device_iterable =
    std::contiguous_iterator<decltype(begin(std::declval<const Container&>()))>;
}  // namespace Algorithm_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Utility functions wrapping STL algorithms and additional algorithms.
 *
 * All functions except `alg::sample` and `alg::sort` can be called in Kokkos
 * kernels on containers with contiguous iterators, such as `std::array` and
 * `Kokkos::Array`. For these containers they are implemented as plain loops
 * rather than by calling the standard library, because the standard library
 * algorithms aren't supported in device code (see \ref gpu_support "GPU
 * support"). To sort short lists on the device, use `alg::bubble_sort`. For
 * other containers, e.g. `std::unordered_map` or `DataVector`, the functions
 * can only be called on the host.
 */
namespace alg {
/// Like `std::all_of` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr bool all_of(const Container& c,
                                      UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  std::decay_t<UnaryPredicate> predicate(
      std::forward<UnaryPredicate>(unary_predicate));
  for (auto it = begin(c); it != end(c); ++it) {
    if (not predicate(*it)) {
      return false;
    }
  }
  return true;
}

/// \cond
template <class Container, class UnaryPredicate>
bool all_of(const Container& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return std::all_of(begin(c), end(c),
                     std::forward<UnaryPredicate>(unary_predicate));
}
/// \endcond

/// Like `std::any_of` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr bool any_of(const Container& c,
                                      UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  std::decay_t<UnaryPredicate> predicate(
      std::forward<UnaryPredicate>(unary_predicate));
  for (auto it = begin(c); it != end(c); ++it) {
    if (predicate(*it)) {
      return true;
    }
  }
  return false;
}

/// \cond
template <class Container, class UnaryPredicate>
bool any_of(const Container& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return std::any_of(begin(c), end(c),
                     std::forward<UnaryPredicate>(unary_predicate));
}
/// \endcond

/// Like `std::none_of` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr bool none_of(const Container& c,
                                       UnaryPredicate&& unary_predicate) {
  return not alg::any_of(c, std::forward<UnaryPredicate>(unary_predicate));
}

/// \cond
template <class Container, class UnaryPredicate>
bool none_of(const Container& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return std::none_of(begin(c), end(c),
                      std::forward<UnaryPredicate>(unary_predicate));
}
/// \endcond

/// Like `std::count` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class T>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr auto count(const Container& c, const T& value) {
  using std::begin;
  using std::end;
  typename std::iterator_traits<decltype(begin(c))>::difference_type result = 0;
  for (auto it = begin(c); it != end(c); ++it) {
    if (*it == value) {
      ++result;
    }
  }
  return result;
}

/// \cond
template <class Container, class T>
auto count(const Container& c, const T& value) {
  using std::begin;
  using std::end;
  return std::count(begin(c), end(c), value);
}
/// \endcond

/// Like `std::count_if` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr auto count_if(const Container& c,
                                        UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  std::decay_t<UnaryPredicate> predicate(
      std::forward<UnaryPredicate>(unary_predicate));
  typename std::iterator_traits<decltype(begin(c))>::difference_type result = 0;
  for (auto it = begin(c); it != end(c); ++it) {
    if (predicate(*it)) {
      ++result;
    }
  }
  return result;
}

/// \cond
template <class Container, class UnaryPredicate>
auto count_if(const Container& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return std::count_if(begin(c), end(c),
                       std::forward<UnaryPredicate>(unary_predicate));
}
/// \endcond

/// Convenience wrapper around constexpr reimplementation of std::find
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class T>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) find(Container&& c, const T& value) {
  using std::begin;
  using std::end;
  return cpp20::find(begin(std::forward<Container>(c)),
                     end(std::forward<Container>(c)), value);
}

/// \cond
template <class Container, class T>
constexpr decltype(auto) find(Container&& c, const T& value) {
  using std::begin;
  using std::end;
  return cpp20::find(begin(std::forward<Container>(c)),
                     end(std::forward<Container>(c)), value);
}
/// \endcond

/// Convenience wrapper around constexpr reimplementation of std::find_if
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) find_if(
    Container&& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return cpp20::find_if(begin(std::forward<Container>(c)),
                        end(std::forward<Container>(c)),
                        std::forward<UnaryPredicate>(unary_predicate));
}

/// \cond
template <class Container, class UnaryPredicate>
constexpr decltype(auto) find_if(Container&& c,
                                 UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return cpp20::find_if(begin(std::forward<Container>(c)),
                        end(std::forward<Container>(c)),
                        std::forward<UnaryPredicate>(unary_predicate));
}
/// \endcond

/// Convenience wrapper around constexpr reimplementation of std::find_if_not
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) find_if_not(
    Container&& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return cpp20::find_if_not(begin(std::forward<Container>(c)),
                            end(std::forward<Container>(c)),
                            std::forward<UnaryPredicate>(unary_predicate));
}

/// \cond
template <class Container, class UnaryPredicate>
constexpr decltype(auto) find_if_not(Container&& c,
                                     UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return cpp20::find_if_not(begin(std::forward<Container>(c)),
                            end(std::forward<Container>(c)),
                            std::forward<UnaryPredicate>(unary_predicate));
}
/// \endcond

/// Convenience wrapper around constexpr reimplementation of std::find, returns
/// `true` if `value` is in `c`.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class T>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr bool found(const Container& c, const T& value) {
  using std::begin;
  using std::end;
  return cpp20::find(begin(c), end(c), value) != end(c);
}

/// \cond
template <class Container, class T>
constexpr bool found(const Container& c, const T& value) {
  using std::begin;
  using std::end;
  return cpp20::find(begin(c), end(c), value) != end(c);
}
/// \endcond

/// Convenience wrapper around constexpr reimplementation of std::find_if,
/// returns `true` if the result of `cpp20::find_if` is not equal to `end(c)`.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr bool found_if(const Container& c,
                                        UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return cpp20::find_if(begin(c), end(c),
                        std::forward<UnaryPredicate>(unary_predicate)) !=
         end(c);
}

/// \cond
template <class Container, class UnaryPredicate>
constexpr bool found_if(const Container& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return cpp20::find_if(begin(c), end(c),
                        std::forward<UnaryPredicate>(unary_predicate)) !=
         end(c);
}
/// \endcond

/// Convenience wrapper around constexpr reimplementation of std::find_if_not,
/// returns `true` if the result of `cpp20::find_if_not` is not equal to
/// `end(c)`.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr bool found_if_not(const Container& c,
                                            UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return cpp20::find_if_not(begin(c), end(c),
                            std::forward<UnaryPredicate>(unary_predicate)) !=
         end(c);
}

/// \cond
template <class Container, class UnaryPredicate>
constexpr bool found_if_not(const Container& c,
                            UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return cpp20::find_if_not(begin(c), end(c),
                            std::forward<UnaryPredicate>(unary_predicate)) !=
         end(c);
}
/// \endcond

/// @{
/// Like `std::for_each` on the whole container, returns `f`.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryFunction>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr std::decay_t<UnaryFunction> for_each(
    const Container& c, UnaryFunction&& f) {
  using std::begin;
  using std::end;
  std::decay_t<UnaryFunction> function(std::forward<UnaryFunction>(f));
  for (auto it = begin(c); it != end(c); ++it) {
    function(*it);
  }
  return function;
}

/// \cond
template <class Container, class UnaryFunction>
std::decay_t<UnaryFunction> for_each(const Container& c, UnaryFunction&& f) {
  using std::begin;
  using std::end;
  return std::for_each(begin(c), end(c), std::forward<UnaryFunction>(f));
}
/// \endcond

SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryFunction>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr std::decay_t<UnaryFunction> for_each(
    Container& c, UnaryFunction&& f) {
  using std::begin;
  using std::end;
  std::decay_t<UnaryFunction> function(std::forward<UnaryFunction>(f));
  for (auto it = begin(c); it != end(c); ++it) {
    function(*it);
  }
  return function;
}

/// \cond
template <class Container, class UnaryFunction>
std::decay_t<UnaryFunction> for_each(Container& c, UnaryFunction&& f) {
  using std::begin;
  using std::end;
  return std::for_each(begin(c), end(c), std::forward<UnaryFunction>(f));
}
/// \endcond
/// @}

/// Like `std::equal` on the whole container `lhs`, assumes `rhs` has at least
/// as many elements as `lhs`.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class Container2>
  requires Algorithm_detail::device_iterable<Container> and
           Algorithm_detail::device_iterable<Container2>
KOKKOS_FUNCTION constexpr bool equal(const Container& lhs,
                                     const Container2& rhs) {
  using std::begin;
  using std::end;
  auto rhs_it = begin(rhs);
  for (auto it = begin(lhs); it != end(lhs); ++it, ++rhs_it) {
    if (not(*it == *rhs_it)) {
      return false;
    }
  }
  return true;
}

/// \cond
template <class Container, class Container2>
bool equal(const Container& lhs, const Container2& rhs) {
  using std::begin;
  using std::end;
  return std::equal(begin(lhs), end(lhs), begin(rhs));
}
/// \endcond

/// Like `std::equal` on the whole container `lhs`, assumes `rhs` has at least
/// as many elements as `lhs`.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class Container2, class BinaryPredicate>
  requires Algorithm_detail::device_iterable<Container> and
           Algorithm_detail::device_iterable<Container2>
KOKKOS_FUNCTION constexpr bool equal(const Container& lhs,
                                     const Container2& rhs,
                                     BinaryPredicate&& p) {
  using std::begin;
  using std::end;
  std::decay_t<BinaryPredicate> predicate(std::forward<BinaryPredicate>(p));
  auto rhs_it = begin(rhs);
  for (auto it = begin(lhs); it != end(lhs); ++it, ++rhs_it) {
    if (not predicate(*it, *rhs_it)) {
      return false;
    }
  }
  return true;
}

/// \cond
template <class Container, class Container2, class BinaryPredicate>
bool equal(const Container& lhs, const Container2& rhs, BinaryPredicate&& p) {
  using std::begin;
  using std::end;
  return std::equal(begin(lhs), end(lhs), begin(rhs),
                    std::forward<BinaryPredicate>(p));
}
/// \endcond

/// Like `std::max_element` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class Compare>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) max_element(const Container& c,
                                                     Compare&& comp) {
  using std::begin;
  using std::end;
  std::decay_t<Compare> compare(std::forward<Compare>(comp));
  auto largest = begin(c);
  if (largest != end(c)) {
    auto it = largest;
    for (++it; it != end(c); ++it) {
      if (compare(*largest, *it)) {
        largest = it;
      }
    }
  }
  return largest;
}

/// \cond
template <class Container, class Compare>
constexpr decltype(auto) max_element(const Container& c, Compare&& comp) {
  using std::begin;
  using std::end;
  return std::max_element(begin(c), end(c), std::forward<Compare>(comp));
}
/// \endcond

/// Like `std::max_element` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) max_element(const Container& c) {
  return alg::max_element(c,
                          [](const auto& a, const auto& b) { return a < b; });
}

/// \cond
template <class Container>
constexpr decltype(auto) max_element(const Container& c) {
  using std::begin;
  using std::end;
  return std::max_element(begin(c), end(c));
}
/// \endcond

/// Like `std::min_element` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class Compare>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) min_element(const Container& c,
                                                     Compare&& comp) {
  using std::begin;
  using std::end;
  std::decay_t<Compare> compare(std::forward<Compare>(comp));
  auto smallest = begin(c);
  if (smallest != end(c)) {
    auto it = smallest;
    for (++it; it != end(c); ++it) {
      if (compare(*it, *smallest)) {
        smallest = it;
      }
    }
  }
  return smallest;
}

/// \cond
template <class Container, class Compare>
constexpr decltype(auto) min_element(const Container& c, Compare&& comp) {
  using std::begin;
  using std::end;
  return std::min_element(begin(c), end(c), std::forward<Compare>(comp));
}
/// \endcond

/// Like `std::min_element` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) min_element(const Container& c) {
  return alg::min_element(c,
                          [](const auto& a, const auto& b) { return a < b; });
}

/// \cond
template <class Container>
constexpr decltype(auto) min_element(const Container& c) {
  using std::begin;
  using std::end;
  return std::min_element(begin(c), end(c));
}
/// \endcond

/// Like `std::minmax_element` on the whole container, which returns the first
/// smallest and the last largest element.
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class Compare>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) minmax_element(const Container& c,
                                                        Compare&& comp) {
  using std::begin;
  using std::end;
  std::decay_t<Compare> compare(std::forward<Compare>(comp));
  auto smallest = begin(c);
  auto largest = begin(c);
  if (smallest != end(c)) {
    auto it = smallest;
    for (++it; it != end(c); ++it) {
      if (compare(*it, *smallest)) {
        smallest = it;
      }
      if (not compare(*it, *largest)) {
        largest = it;
      }
    }
  }
  return std::pair{smallest, largest};
}

/// \cond
template <class Container, class Compare>
constexpr decltype(auto) minmax_element(const Container& c, Compare&& comp) {
  using std::begin;
  using std::end;
  return std::minmax_element(begin(c), end(c), std::forward<Compare>(comp));
}
/// \endcond

/// Like `std::minmax_element` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) minmax_element(const Container& c) {
  return alg::minmax_element(
      c, [](const auto& a, const auto& b) { return a < b; });
}

/// \cond
template <class Container>
constexpr decltype(auto) minmax_element(const Container& c) {
  using std::begin;
  using std::end;
  return std::minmax_element(begin(c), end(c));
}
/// \endcond

/// Like `std::remove_if` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class UnaryPredicate>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) remove_if(
    Container& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  std::decay_t<UnaryPredicate> predicate(
      std::forward<UnaryPredicate>(unary_predicate));
  auto result = begin(c);
  for (auto it = begin(c); it != end(c); ++it) {
    if (not predicate(*it)) {
      if (result != it) {
        *result = std::move(*it);
      }
      ++result;
    }
  }
  return result;
}

/// \cond
template <class Container, class UnaryPredicate>
decltype(auto) remove_if(Container& c, UnaryPredicate&& unary_predicate) {
  using std::begin;
  using std::end;
  return std::remove_if(begin(c), end(c),
                        std::forward<UnaryPredicate>(unary_predicate));
}
/// \endcond

/// Like `std::remove` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class T>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr decltype(auto) remove(Container& c, const T& value) {
  return alg::remove_if(c, [&value](const auto& x) { return x == value; });
}

/// \cond
template <class Container, class T>
decltype(auto) remove(Container& c, const T& value) {
  using std::begin;
  using std::end;
  return std::remove(begin(c), end(c), value);
}
/// \endcond

/// Convience wrapper around std::sample. This can't be called in Kokkos
/// kernels.
template <class Container, class OutputIt, class Distance, class URBG>
decltype(auto) sample(Container& c, OutputIt out, Distance n, URBG&& g) {
  using std::begin;
  using std::end;
  return std::sample(begin(c), end(c), out, n, g);
}

/// @{
/*!
 * \brief Sorts `c` with a bubble sort.
 *
 * Use this to sort short lists on the device, where `alg::sort` can't be used.
 * It takes \f$O(N^2)\f$ operations, so it is only suitable for short lists.
 * Elements that compare equal keep their order.
 */
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class Compare>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr void bubble_sort(Container& c, Compare comp) {
  using std::begin;
  using std::end;
  const auto first = begin(c);
  auto last = end(c);
  // After each pass the largest remaining element is in its place at the end
  bool swapped = true;
  while (swapped and last != first) {
    swapped = false;
    for (auto it = first; it + 1 != last; ++it) {
      if (comp(*(it + 1), *it)) {
        cpp20::iter_swap(it, it + 1);
        swapped = true;
      }
    }
    --last;
  }
}

SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr void bubble_sort(Container& c) {
  alg::bubble_sort(c, [](const auto& a, const auto& b) { return a < b; });
}
/// @}

/// @{
/*!
 * \brief Like `std::sort` on the whole container.
 *
 * This can't be called on the device, where it aborts. Use `alg::bubble_sort`
 * to sort short lists on the device.
 */
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class Compare>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr void sort(Container& c, Compare&& comp) {
  // Device code that calls `alg::sort` compiles, so that the host-only uses
  // still compile in GPU builds (they are also compiled for the device), but
  // it aborts at runtime
  KOKKOS_IF_ON_HOST((using std::begin; using std::end;
                     std::sort(begin(c), end(c), std::forward<Compare>(comp));))
  KOKKOS_IF_ON_DEVICE(
      (static_cast<void>(c); static_cast<void>(comp);
       Kokkos::abort("alg::sort can't be called on the device. Use "
                     "alg::bubble_sort to sort short lists on the device.");))
}

/// \cond
template <class Container, class Compare>
void sort(Container& c, Compare&& comp) {
  using std::begin;
  using std::end;
  std::sort(begin(c), end(c), std::forward<Compare>(comp));
}
/// \endcond

SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr void sort(Container& c) {
  alg::sort(c, [](const auto& a, const auto& b) { return a < b; });
}

/// \cond
template <class Container>
void sort(Container& c) {
  using std::begin;
  using std::end;
  std::sort(begin(c), end(c));
}
/// \endcond
/// @}

/// Like `std::transform` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container, class OutputIt, class UnaryOp>
  requires Algorithm_detail::device_iterable<Container>
KOKKOS_FUNCTION constexpr OutputIt transform(const Container& c,
                                             OutputIt output_first,
                                             UnaryOp&& unary_op) {
  using std::begin;
  using std::end;
  std::decay_t<UnaryOp> op(std::forward<UnaryOp>(unary_op));
  for (auto it = begin(c); it != end(c); ++it, ++output_first) {
    *output_first = op(*it);
  }
  return output_first;
}

/// \cond
template <class Container, class OutputIt, class UnaryOp>
OutputIt transform(const Container& c, OutputIt output_first,
                   UnaryOp&& unary_op) {
  using std::begin;
  using std::end;
  return std::transform(begin(c), end(c), output_first,
                        std::forward<UnaryOp>(unary_op));
}
/// \endcond

/// Like `std::transform` on the whole container
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <class Container1, class Container2, class OutputIt, class BinaryOp>
  requires Algorithm_detail::device_iterable<Container1> and
           Algorithm_detail::device_iterable<Container2>
KOKKOS_FUNCTION constexpr OutputIt transform(const Container1& c1,
                                             const Container2& c2,
                                             OutputIt output_first,
                                             BinaryOp&& binary_op) {
  using std::begin;
  using std::end;
  std::decay_t<BinaryOp> op(std::forward<BinaryOp>(binary_op));
  auto it2 = begin(c2);
  for (auto it1 = begin(c1); it1 != end(c1); ++it1, ++it2, ++output_first) {
    *output_first = op(*it1, *it2);
  }
  return output_first;
}

/// \cond
template <class Container1, class Container2, class OutputIt, class BinaryOp>
OutputIt transform(const Container1& c1, const Container2& c2,
                   OutputIt output_first, BinaryOp&& binary_op) {
  using std::begin;
  using std::end;
  return std::transform(begin(c1), end(c1), begin(c2), output_first,
                        std::forward<BinaryOp>(binary_op));
}
/// \endcond
}  // namespace alg
