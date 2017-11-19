// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function divide_by

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

/*!
 * \ingroup TensorGroup
 * \brief Divides the components of a tensor by a scalar
 *
 * \returns a tensor of the same type as the input tensor
 */
template <typename TensorType>
TensorType divide_by(TensorType tensor, const DataVector& divisor) {
  ASSERT(tensor.get(0).size() == divisor.size(),
         "The DataVectors in `tensor` and `divisor` do not have the same size");
  for (auto& component : tensor) {
    component /= divisor;
  }
  return tensor;
}
