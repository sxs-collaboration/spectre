// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::TensorExpressions {

/// \ingroup TestingFrameworkGroup
/// \brief Test that the computed tensor multi-index of a rank 0 RHS Tensor is
/// equivalent to the given LHS tensor multi-index
///
/// \tparam DataType the type of data being stored in the Tensors
template <typename DataType>
void test_compute_rhs_tensor_index_rank_0() noexcept {
  const Tensor<DataType> rhs_tensor{};
  // Get TensorExpression from RHS tensor
  const auto R = rhs_tensor();

  const std::array<size_t, 0> index_order = {};

  const std::array<size_t, 0> tensor_multi_index = {};
  CHECK(R.compute_rhs_tensor_index(index_order, index_order,
                                   tensor_multi_index) == tensor_multi_index);
}

}  // namespace TestHelpers::TensorExpressions
