// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \ingroup TensorGroup
/// \brief returns the Identity matrix
template <size_t Dim, typename DataType>
tnsr::Ij<DataType, Dim, Frame::NoFrame> identity(
    const DataType& used_for_type) noexcept {
  auto identity_matrix{make_with_value<tnsr::Ij<DataType, Dim, Frame::NoFrame>>(
      used_for_type, 0.0)};

  for (size_t i = 0; i < Dim; ++i) {
    identity_matrix.get(i, i) = 1.0;
  }
  return identity_matrix;
}
