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
  TestHelpers::tenex::test_tensor_index_transformation_rank_1(ti::k);

  // Rank 2
  TestHelpers::tenex::test_tensor_index_transformation_rank_2(ti::i, ti::j);
  TestHelpers::tenex::test_tensor_index_transformation_rank_2(ti::j, ti::i);

  // Rank 3
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti::a, ti::b,
                                                              ti::c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti::a, ti::c,
                                                              ti::b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti::b, ti::a,
                                                              ti::c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti::b, ti::c,
                                                              ti::a);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti::c, ti::a,
                                                              ti::b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_3(ti::c, ti::b,
                                                              ti::a);

  // Rank 4
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::a, ti::b,
                                                              ti::c, ti::d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::a, ti::b,
                                                              ti::d, ti::c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::a, ti::c,
                                                              ti::b, ti::d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::a, ti::c,
                                                              ti::d, ti::b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::a, ti::d,
                                                              ti::b, ti::c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::a, ti::d,
                                                              ti::c, ti::b);

  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::b, ti::a,
                                                              ti::c, ti::d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::b, ti::a,
                                                              ti::c, ti::d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::b, ti::c,
                                                              ti::a, ti::d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::b, ti::c,
                                                              ti::d, ti::a);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::b, ti::d,
                                                              ti::a, ti::c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::b, ti::d,
                                                              ti::c, ti::a);

  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::c, ti::a,
                                                              ti::b, ti::d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::c, ti::a,
                                                              ti::d, ti::b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::c, ti::b,
                                                              ti::a, ti::d);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::c, ti::b,
                                                              ti::d, ti::a);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::c, ti::d,
                                                              ti::a, ti::b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::c, ti::d,
                                                              ti::b, ti::a);

  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::d, ti::a,
                                                              ti::b, ti::c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::d, ti::a,
                                                              ti::c, ti::b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::d, ti::b,
                                                              ti::a, ti::c);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::d, ti::b,
                                                              ti::c, ti::a);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::d, ti::c,
                                                              ti::a, ti::b);
  TestHelpers::tenex::test_tensor_index_transformation_rank_4(ti::d, ti::c,
                                                              ti::b, ti::a);

  // Test transformations involving time indices
  TestHelpers::tenex::test_tensor_index_transformation_with_time_indices();
}
