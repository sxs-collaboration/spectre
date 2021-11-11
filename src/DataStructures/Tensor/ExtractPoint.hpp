// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

/// \ingroup DataStructuresGroup
/// Copy a given index of each component of a `Tensor<DataVector>`
/// into a `Tensor<double>`.
template <typename... Structure>
Tensor<double, Structure...> extract_point(
    const Tensor<DataVector, Structure...>& tensor, const size_t index) {
  Tensor<double, Structure...> result;
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = tensor[i][index];
  }
  return result;
}
