// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/*!
 * \ingroup UtilitiesGroup
 * \brief Get the indices in the container `t` of the `K` elements that satisfy
 * `op` after sorting with respect to `op`.
 *
 * Specifically, if `op` is `[](const lhs, const rhs) { return lhs < rhs; }`
 * then the result will be the indices of the `K` smallest elements in `t`. If
 * `op` is `[](const lhs, const rhs) { return lhs > rhs; }` then the result will
 * be the indices of the `K` largest elements in `t`.
 *
 * \warning The indices are not sorted by `op` when returned. Use
 * `sorted_indices_of`.
 */
template <size_t K, typename T, typename Comparator>
auto indices_of(const T& t, Comparator op) noexcept -> std::array<size_t, K> {
  ASSERT(K <= t.size(),
         "The container whose K elements we're finding must have at least size "
             << K << " but has size " << t.size());
  std::array<size_t, K> indices{};
  std::iota(indices.begin(), indices.end(), 0);
  // Each value in `indices` is an index into `t`. get_index_of_max
  // returns the index into `indices` corresponding to the largest (per
  // comparison `op`) value of `t`.
  const auto get_index_of_max = [&indices, &t, &op ]() noexcept {
    return static_cast<size_t>(std::distance(
        indices.begin(), std::max_element(indices.begin(), indices.end(), [
          &t, &op
        ](const size_t lhs, const size_t rhs) noexcept {
          return op(gsl::at(t, lhs), gsl::at(t, rhs));
        })));
  };
  auto index_of_max = get_index_of_max();
  for (size_t i = K; i < t.size(); ++i) {
    if (op(gsl::at(t, i), gsl::at(t, gsl::at(indices, index_of_max)))) {
      gsl::at(indices, index_of_max) = i;
      index_of_max = get_index_of_max();
    }
  }
  return indices;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Same as `indices_of` but returns the list of indices sorted by `op`.
 */
template <size_t K, typename T, typename Comparator>
auto sorted_indices_of(const T& t, Comparator op) noexcept
    -> std::array<size_t, K> {
  auto result = indices_of<K>(t, op);
  std::sort(result.begin(), result.end(),
            [&t, &op](const size_t lhs, const size_t rhs) {
              return op(gsl::at(t, lhs), gsl::at(t, rhs));
            });
  return result;
}
