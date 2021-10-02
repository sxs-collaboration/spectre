// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"

namespace TestHelpers::TensorExpressions {
/// \ingroup TestingFrameworkGroup
/// \brief Test that the transformation between two rank 0 tensors' generic
/// indices and the subsequent transformed multi-index is correctly computed
///
/// \details The functions tested are:
/// - `TensorExpressions::compute_tensorindex_transformation`
/// - `TensorExpressions::transform_multi_index`
void test_tensor_index_transformation_rank_0() {
  const std::array<size_t, 0> index_order = {};

  const std::array<size_t, 0> actual_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order,
                                                              index_order);
  const std::array<size_t, 0> expected_transformation = {};
  CHECK(actual_transformation == expected_transformation);

  const std::array<size_t, 0> tensor_multi_index = {};
  CHECK(::TensorExpressions::transform_multi_index(
            tensor_multi_index, expected_transformation) == tensor_multi_index);
}
}  // namespace TestHelpers::TensorExpressions
