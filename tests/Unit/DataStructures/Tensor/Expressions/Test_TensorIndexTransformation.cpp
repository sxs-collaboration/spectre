// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank0TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank1TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank2TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank3TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank4TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationTimeIndex.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.TensorIndexTransformation",
    "[DataStructures][Unit]") {
  // Rank 0
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_0();

  // Rank 1
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_1(ti_k);

  // Rank 2
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_2(ti_i,
                                                                          ti_j);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_2(ti_j,
                                                                          ti_i);

  // Rank 3
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_3(
      ti_a, ti_b, ti_c);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_3(
      ti_a, ti_c, ti_b);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_3(
      ti_b, ti_a, ti_c);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_3(
      ti_b, ti_c, ti_a);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_3(
      ti_c, ti_a, ti_b);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_3(
      ti_c, ti_b, ti_a);

  // Rank 4
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_a, ti_b, ti_c, ti_d);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_a, ti_b, ti_d, ti_c);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_a, ti_c, ti_b, ti_d);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_a, ti_c, ti_d, ti_b);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_a, ti_d, ti_b, ti_c);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_a, ti_d, ti_c, ti_b);

  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_b, ti_a, ti_c, ti_d);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_b, ti_a, ti_c, ti_d);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_b, ti_c, ti_a, ti_d);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_b, ti_c, ti_d, ti_a);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_b, ti_d, ti_a, ti_c);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_b, ti_d, ti_c, ti_a);

  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_c, ti_a, ti_b, ti_d);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_c, ti_a, ti_d, ti_b);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_c, ti_b, ti_a, ti_d);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_c, ti_b, ti_d, ti_a);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_c, ti_d, ti_a, ti_b);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_c, ti_d, ti_b, ti_a);

  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_d, ti_a, ti_b, ti_c);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_d, ti_a, ti_c, ti_b);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_d, ti_b, ti_a, ti_c);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_d, ti_b, ti_c, ti_a);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_d, ti_c, ti_a, ti_b);
  TestHelpers::TensorExpressions::test_tensor_index_transformation_rank_4(
      ti_d, ti_c, ti_b, ti_a);

  // Test transformations involving time indices
  TestHelpers::TensorExpressions::
      test_tensor_index_transformation_with_time_indices();
}
