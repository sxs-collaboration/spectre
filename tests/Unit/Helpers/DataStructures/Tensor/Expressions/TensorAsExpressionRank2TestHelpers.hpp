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
/// the subsequent computed RHS multi-index of a rank 2 tensor is correctly
/// computed by the functions of TensorAsExpression, according to the orders of
/// the LHS and RHS generic indices
///
/// \details The functions tested are:
/// - `TensorAsExpression::compute_index_transformation`
/// - `TensorAsExpression::compute_rhs_multi_index`
///
/// If we consider the RHS tensor's generic indices to be (a, b), the possible
/// orderings of the LHS tensor's generic indices are: (a, b) and (b, a). For
/// each of these cases, this test checks that for each LHS component's
/// multi-index, the equivalent RHS multi-index is correctly computed.
///
/// \param tensorindex_a the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_a`
/// \param tensorindex_b the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti_B`
template <typename TensorIndexA, typename TensorIndexB>
void test_tensor_as_expression_rank_2(
    const TensorIndexA& tensorindex_a,
    const TensorIndexB& tensorindex_b) noexcept {
  const size_t dim_a = 3;
  const size_t dim_b = 4;

  const IndexType indextype_a =
      TensorIndexA::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;
  const IndexType indextype_b =
      TensorIndexB::is_spacetime ? IndexType::Spacetime : IndexType::Spatial;

  Tensor<
      double, Symmetry<2, 1>,
      index_list<Tensor_detail::TensorIndexType<dim_a, TensorIndexA::valence,
                                                Frame::Inertial, indextype_a>,
                 Tensor_detail::TensorIndexType<dim_b, TensorIndexB::valence,
                                                Frame::Inertial, indextype_b>>>
      rhs_tensor{};
  std::iota(rhs_tensor.begin(), rhs_tensor.end(), 0.0);
  // Get TensorExpression from RHS tensor
  const auto R_ab_expr = rhs_tensor(tensorindex_a, tensorindex_b);

  const std::array<size_t, 2> index_order_ab = {TensorIndexA::value,
                                                TensorIndexB::value};
  const std::array<size_t, 2> index_order_ba = {TensorIndexB::value,
                                                TensorIndexA::value};

  const std::array<size_t, 2> actual_ab_to_ab_transformation =
      R_ab_expr.compute_index_transformation(index_order_ab);
  const std::array<size_t, 2> expected_ab_to_ab_transformation = {0, 1};
  const std::array<size_t, 2> actual_ba_to_ab_transformation =
      R_ab_expr.compute_index_transformation(index_order_ba);
  const std::array<size_t, 2> expected_ba_to_ab_transformation = {1, 0};

  CHECK(actual_ab_to_ab_transformation == expected_ab_to_ab_transformation);
  CHECK(actual_ba_to_ab_transformation == expected_ba_to_ab_transformation);

  for (size_t i = 0; i < dim_a; i++) {
    for (size_t j = 0; j < dim_b; j++) {
      const std::array<size_t, 2> ij = {i, j};
      const std::array<size_t, 2> ji = {j, i};

      // For L_{ab} = R_{ab}, check that L_{ij} == R_{ij}
      CHECK(R_ab_expr.compute_rhs_multi_index(
                ij, expected_ab_to_ab_transformation) == ij);
      // For L_{ba} = R_{ab}, check that L_{ij} == R_{ji}
      CHECK(R_ab_expr.compute_rhs_multi_index(
                ij, expected_ba_to_ab_transformation) == ji);
    }
  }
}
}  // namespace TestHelpers::TensorExpressions
