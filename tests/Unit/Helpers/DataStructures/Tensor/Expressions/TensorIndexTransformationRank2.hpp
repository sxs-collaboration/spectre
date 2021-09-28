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
/// \brief Test that the transformation between two rank 2 tensors' generic
/// indices and the subsequent transformed multi-indices are correctly computed
///
/// \details The functions tested are:
/// - `TensorExpressions::compute_tensorindex_transformation`
/// - `TensorExpressions::transform_multi_index`
///
/// If we consider the first tensor's generic indices to be (a, b), the possible
/// orderings of the second tensor's generic indices are: (a, b) and (b, a). For
/// each of these cases, this test checks that for each multi-index with the
/// first generic index ordering, the equivalent multi-index with the second
/// ordering is correctly computed.
///
/// \tparam TensorIndexA the first generic tensor index, e.g. type of `ti_a`
/// \tparam TensorIndexB the second generic tensor index, e.g. type of `ti_B`
template <typename TensorIndexA, typename TensorIndexB>
void test_tensor_index_transformation_rank_2(
    const TensorIndexA& /*tensorindex_a*/,
    const TensorIndexB& /*tensorindex_b*/) {
  const size_t dim_a = 3;
  const size_t dim_b = 4;

  const std::array<size_t, 2> index_order_ab = {TensorIndexA::value,
                                                TensorIndexB::value};
  const std::array<size_t, 2> index_order_ba = {TensorIndexB::value,
                                                TensorIndexA::value};

  const std::array<size_t, 2> actual_ab_to_ab_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_ab,
                                                              index_order_ab);
  const std::array<size_t, 2> expected_ab_to_ab_transformation = {0, 1};
  const std::array<size_t, 2> actual_ba_to_ab_transformation =
      ::TensorExpressions::compute_tensorindex_transformation(index_order_ba,
                                                              index_order_ab);
  const std::array<size_t, 2> expected_ba_to_ab_transformation = {1, 0};

  CHECK(actual_ab_to_ab_transformation == expected_ab_to_ab_transformation);
  CHECK(actual_ba_to_ab_transformation == expected_ba_to_ab_transformation);

  for (size_t i = 0; i < dim_a; i++) {
    for (size_t j = 0; j < dim_b; j++) {
      const std::array<size_t, 2> ij = {i, j};
      const std::array<size_t, 2> ji = {j, i};

      // For L_{ab} = R_{ab}, check that L_{ij} == R_{ij}
      CHECK(::TensorExpressions::transform_multi_index(
                ij, expected_ab_to_ab_transformation) == ij);
      // For L_{ba} = R_{ab}, check that L_{ij} == R_{ji}
      CHECK(::TensorExpressions::transform_multi_index(
                ij, expected_ba_to_ab_transformation) == ji);
    }
  }
}
}  // namespace TestHelpers::TensorExpressions
