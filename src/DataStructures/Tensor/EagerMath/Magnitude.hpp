// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions euclidean_magnitude and magnitude

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

/*!
 * \ingroup TensorGroup
 * \brief Compute the Euclidean magnitude of a rank-1 tensor
 *
 * \details
 * Computes the square root of the sum of the squares of the components of
 * the rank-1 tensor.
 */
template <typename DataType, typename Index>
Scalar<DataType> magnitude(
    const Tensor<DataType, Symmetry<1>, tmpl::list<Index>>& vector) noexcept {
  auto magnitude_squared = make_with_value<DataType>(vector, 0.);
  for (size_t d = 0; d < Index::dim; ++d) {
    magnitude_squared += square(vector.get(d));
  }
  return Scalar<DataType>{sqrt(magnitude_squared)};
}

/*!
 * \ingroup TensorGroup
 * \brief Compute the magnitude of a rank-1 tensor
 *
 * \details
 * Returns the square root of the input tensor contracted twice with the given
 * metric.
 */
template <typename DataType, typename Index>
Scalar<DataType> magnitude(
    const Tensor<DataType, Symmetry<1>, tmpl::list<Index>>& vector,
    const Tensor<DataType, Symmetry<1, 1>,
                 tmpl::list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>&
        metric) noexcept {
  auto magnitude_squared = make_with_value<DataType>(vector, 0.);
  for (size_t a = 0; a < Index::dim; ++a) {
    for (size_t b = 0; b < Index::dim; ++b) {
      magnitude_squared += vector.get(a) * vector.get(b) * metric.get(a, b);
    }
  }
  return Scalar<DataType>{sqrt(magnitude_squared)};
}
