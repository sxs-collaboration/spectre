// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace TestHelpers::TensorExpressions {
/// \ingroup TestingFrameworkGroup
/// \brief Test that the transformation between LHS and RHS multi-indices and
/// the subsequent computed RHS multi-index of a rank 1 tensor is correctly
/// computed by the functions of TensorAsExpression
///
/// \details The functions tested are:
/// - `TensorAsExpression::compute_index_transformation`
/// - `TensorAsExpression::compute_rhs_multi_index`
///
/// \param tensorindex the TensorIndex used in the the TensorExpression,
/// e.g. `ti_a`
template <typename TensorIndex>
void test_tensor_as_expression_rank_1(const TensorIndex& tensorindex) noexcept {
  const size_t dim = 3;
  const IndexType indextype =
      TensorIndex::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;

  Tensor<double, Symmetry<1>,
         index_list<Tensor_detail::TensorIndexType<dim, TensorIndex::valence,
                                                   Frame::Grid, indextype>>>
      rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);
  // Get TensorExpression from RHS tensor
  const auto R_a_expr = rhs_tensor(tensorindex);

  const std::array<size_t, 1> index_order = {TensorIndex::value};

  const std::array<size_t, 1> actual_transformation =
      R_a_expr.compute_index_transformation(index_order);
  const std::array<size_t, 1> expected_transformation = {0};
  CHECK(actual_transformation == expected_transformation);

  // For L_a = R_a, check that L_i == R_i
  for (size_t i = 0; i < dim; i++) {
    const std::array<size_t, 1> tensor_multi_index = {i};
    CHECK(R_a_expr.compute_rhs_multi_index(tensor_multi_index,
                                           expected_transformation) ==
          tensor_multi_index);
  }
}
}  // namespace TestHelpers::TensorExpressions
