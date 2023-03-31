// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

// This implementation is from:
// https://stackoverflow.com/questions/16686942/how-to-iterate-over-two-stl-like-containers-cartesian-product

namespace detail {

// the lambda is fully bound with one element from each of the ranges
template <class Op>
void insert_tuples(Op op) {
  // evaluating the lambda will insert the currently bound tuple
  op();
}

// "peel off" the first range from the remaining tuple of ranges
template <class Op, class InputIterator1, class... InputIterator2>
void insert_tuples(Op op, std::pair<InputIterator1, InputIterator1> head,
                   std::pair<InputIterator2, InputIterator2>... tail) {
  // "peel off" the elements from the first of the remaining ranges
  // NOTE: the recursion will effectively generate the multiple nested for-loops
  for (auto it = head.first; it != head.second; ++it) {
    // bind the first free variable in the lambda, and
    // keep one free variable for each of the remaining ranges
    detail::insert_tuples(
        [&op, &it](InputIterator2... elems) { op(it, elems...); }, tail...);
  }
}

}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Fill the `result` iterator with the Cartesian product of a sequence of
 * iterators
 *
 * The `result` will hold all possible combinations of the input iterators.
 * The last dimension varies fastest.
 */
template <class OutputIterator, class... InputIterator>
void cartesian_product(OutputIterator result,
                       std::pair<InputIterator, InputIterator>... dimensions) {
  detail::insert_tuples(
      [&result](InputIterator... elems) {
        *result++ = std::make_tuple(*elems...);
      },
      dimensions...);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief The Cartesian product of a sequence of arrays
 *
 * Returns a `std::array` with all possible combinations of the input arrays.
 * The last dimension varies fastest.
 *
 * \example
 * Here's an example using this function to replace a nested for loop:
 * \snippet Test_Wedge3D.cpp cartesian_product_loop
 */
template <typename... Ts, size_t... Lens>
std::array<std::tuple<Ts...>, (... * Lens)> cartesian_product(
    const std::array<Ts, Lens>&... dimensions) {
  std::array<std::tuple<Ts...>, (... * Lens)> result{};
  cartesian_product(result.begin(),
                    std::make_pair(dimensions.begin(), dimensions.end())...);
  return result;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief The Cartesian product of several containers
 *
 * Returns a `std::vector` with all possible combinations of the values of the
 * input containers. The value of the last container varies fastest.
 */
template <typename... Containers>
std::vector<std::tuple<typename Containers::value_type...>> cartesian_product(
    const Containers&... containers) {
  std::vector<std::tuple<typename Containers::value_type...>> result{};
  cartesian_product(
      std::back_inserter(result),
      std::make_pair(std::begin(containers), std::end(containers))...);
  return result;
}
