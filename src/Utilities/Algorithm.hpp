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
 * \brief Compute the indices of the 'K` smallest elements of the container 't'
 *
 * \warning The order of the indices is not guaranteed. That is, the first index
 * is not guaranteed to be the of the smallest element in `t`
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
 * \brief Same as `indices_of` but returns the list of indices sorted such that
 * the order corresponds to the input `t` having been sorted by `op`.
 */
template <size_t K, typename T, typename Comparator>
auto sorted_indices_of(const T& t, Comparator op) noexcept
    -> std::array<size_t, K> {
  auto result = indices_of<K>(t, op);
  std::sort(result.begin(), result.end(),
            [&t, &op ](const size_t lhs, const size_t rhs) noexcept {
              return op(gsl::at(t, lhs), gsl::at(t, rhs));
            });
  return result;
}
