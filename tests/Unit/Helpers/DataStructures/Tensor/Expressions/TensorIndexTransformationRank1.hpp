// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/TensorIndexTransformation.hpp"

namespace TestHelpers::TensorExpressions {
/// \ingroup TestingFrameworkGroup
/// \brief Test that the transformation between two rank 1 tensors' generic
/// indices and the subsequent transformed multi-indices are correctly computed
///
/// \details The functions tested are:
/// - `TensorExpressions::compute_tensorindex_transformation`
/// - `TensorExpressions::transform_multi_index`
///
/// \tparam TensorIndex the first generic tensor index, e.g. type of `ti_a`
template <typename TensorIndex>
void test_tensor_index_transformation_rank_1(
    const TensorIndex& /*tensorindex*/) noexcept {
  const size_t dim = 3;

  const std::array<size_t, 1> index_order = {TensorIndex::value};

  const std::array<size_t, 1> actual_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order,
                                                              index_order);
  const std::array<size_t, 1> expected_transformation = {0};
  CHECK(actual_transformation == expected_transformation);

  // For L_a = R_a, check that L_i == R_i
  for (size_t i = 0; i < dim; i++) {
    const std::array<size_t, 1> tensor_multi_index = {i};
    CHECK(::TensorExpressions::transform_multi_index(tensor_multi_index,
                                                     expected_transformation) ==
          tensor_multi_index);
  }
}
}  // namespace TestHelpers::TensorExpressions
