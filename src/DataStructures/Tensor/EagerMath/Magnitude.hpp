// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions euclidean_magnitude and magnitude

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"

/*!
 * \ingroup TensorGroup
 * \brief compute the Euclidean magnitude of a rank-1 tensor
 *
 * \returns the Euclidean magnitude of the rank-1 tensor
 *
 * \details
 * Computes the square root of the sum of the squares of the components of
 * the rank-1 tensor.
 */
template <typename DataType, typename Index>
DataType magnitude(
    const Tensor<DataType, Symmetry<1>, typelist<Index>>& tensor) {
  auto magnitude_squared = make_with_value<DataType>(tensor, 0.);
  for (size_t d = 0; d < Index::dim; ++d) {
    magnitude_squared += square(tensor.get(d));
  }
  return sqrt(magnitude_squared);
}

/*!
 * \ingroup TensorGroup
 * \brief compute the magnitude of a rank-1 tensor
 *
 * \returns the magnitude of the rank-1 tensor
 *
 * \details
 * Returns the square root of the input tensor contracted twice with the given
 * metric.
 */
template <typename DataType, typename Index0, typename Index1>
DataType magnitude(
    const Tensor<DataType, Symmetry<1>, typelist<Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>, typelist<Index1, Index1>>&
        metric) {
  static_assert(std::is_same<Index0, change_index_up_lo<Index1>>::value,
                "The indices of the tensor and metric must be the same, "
                "except for their valence which must be opposite");
  auto magnitude_squared = make_with_value<DataType>(tensor, 0.);
  for (size_t a = 0; a < Index0::dim; ++a) {
    for (size_t b = 0; b < Index0::dim; ++b) {
      magnitude_squared += tensor.get(a) * tensor.get(b) * metric.get(a, b);
    }
  }
  return sqrt(magnitude_squared);
}
