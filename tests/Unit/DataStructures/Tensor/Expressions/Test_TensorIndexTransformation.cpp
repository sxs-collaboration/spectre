// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank0.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank1.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank3.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationRank4.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TensorIndexTransformationTimeIndex.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.TensorIndexTransformation",
    "[DataStructures][Unit]") {
  // Rank 0
  TestHelpers::tenex::test_tensor_index_transformation_rank_0();

  // Rank 1
  TestHelpers::tenex::test_tensor_index_transformation_rank_1(ti_k);

  // Rank 2
  TestHelpers::tenex::test_tensor_index_transformation_rank_2(ti_i, ti_j);
  TestHelpers::tenex::test_tensor_index_transformation_rank_2(ti_j, ti_i);

  // Rank 3
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti_a, ti_b, ti_c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti_a, ti_c, ti_b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti_b, ti_a, ti_c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti_b, ti_c, ti_a);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti_c, ti_a, ti_b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti_c, ti_b, ti_a);

  // Rank 4
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_a, ti_b, ti_c,
                                                              ti_d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_a, ti_b, ti_d,
                                                              ti_c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_a, ti_c, ti_b,
                                                              ti_d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_a, ti_c, ti_d,
                                                              ti_b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_a, ti_d, ti_b,
                                                              ti_c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_a, ti_d, ti_c,
                                                              ti_b);

  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_b, ti_a, ti_c,
                                                              ti_d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_b, ti_a, ti_c,
                                                              ti_d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_b, ti_c, ti_a,
                                                              ti_d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_b, ti_c, ti_d,
                                                              ti_a);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_b, ti_d, ti_a,
                                                              ti_c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_b, ti_d, ti_c,
                                                              ti_a);

  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_c, ti_a, ti_b,
                                                              ti_d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_c, ti_a, ti_d,
                                                              ti_b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_c, ti_b, ti_a,
                                                              ti_d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_c, ti_b, ti_d,
                                                              ti_a);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_c, ti_d, ti_a,
                                                              ti_b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_c, ti_d, ti_b,
                                                              ti_a);

  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_d, ti_a, ti_b,
                                                              ti_c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_d, ti_a, ti_c,
                                                              ti_b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_d, ti_b, ti_a,
                                                              ti_c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_d, ti_b, ti_c,
                                                              ti_a);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_d, ti_c, ti_a,
                                                              ti_b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti_d, ti_c, ti_b,
                                                              ti_a);

  // Test transformations involving time indices
  TestHelpers::tenex::test_tensor_index_transformation_with_time_indices();
}
