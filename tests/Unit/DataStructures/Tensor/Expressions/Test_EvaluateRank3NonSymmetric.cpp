// Distributed under the MIT License.
// See LICENSE.txt for details.

// Rank 3 test cases for tenex::evaluate are split into this file and
// Test_EvaluateRank3Symmetric.cpp in order to reduce compile time memory usage
// per cpp file.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <IndexType... Is>
using indextype_list = tmpl::integral_list<IndexType, Is...>;

const IndexType spatial_index = IndexType::Spatial;
const IndexType spacetime_index = IndexType::Spacetime;

// \brief Test evaluation of rank 3 tensors with no symmetry
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_rank_3() {
  // spacetime, spacetime, spatial
  TestHelpers::tenex::test_evaluate_rank_3<
      true, DataType, Symmetry<3, 2, 1>,
      indextype_list<spacetime_index, spacetime_index, spatial_index>, ti::d,
      ti::A, ti::i, Frame::Inertial>();

  // spatial, spacetime, spatial
  TestHelpers::tenex::test_evaluate_rank_3<
      true, DataType, Symmetry<3, 2, 1>,
      indextype_list<spatial_index, spacetime_index, spatial_index>, ti::K,
      ti::f, ti::m, Frame::Grid>();
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3NonSymmetric",
    "[DataStructures][Unit]") {
  test_evaluate_rank_3<double>();
  test_evaluate_rank_3<DataVector>();
}
