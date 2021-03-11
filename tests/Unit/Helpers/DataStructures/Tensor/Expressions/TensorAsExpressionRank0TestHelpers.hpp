// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace TestHelpers::TensorExpressions {
/// \ingroup TestingFrameworkGroup
/// \brief Test that the transformation between LHS and RHS multi-indices and
/// the subsequent computed RHS multi-index of a rank 0 tensor is correctly
/// computed by the functions of TensorAsExpression
///
/// \details The functions tested are:
/// - `TensorAsExpression::compute_index_transformation`
/// - `TensorAsExpression::compute_rhs_multi_index`
void test_tensor_as_expression_rank_0() noexcept {
  const Tensor<double> rhs_tensor{{{2.6}}};
  // Get TensorExpression from RHS tensor
  const auto R_expr = rhs_tensor();

  const std::array<size_t, 0> index_order = {};

  const std::array<size_t, 0> actual_transformation =
      R_expr.compute_index_transformation(index_order);
  const std::array<size_t, 0> expected_transformation = {};
  CHECK(actual_transformation == expected_transformation);

  const std::array<size_t, 0> tensor_multi_index = {};
  CHECK(R_expr.compute_rhs_multi_index(
            tensor_multi_index, expected_transformation) == tensor_multi_index);
}
}  // namespace TestHelpers::TensorExpressions
