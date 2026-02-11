// Distributed under the MIT License.
// See LICENSE.txt for details.

// Rank 3 test cases for tenex::evaluate are split into this file and
// Test_EvaluateRank3NonSymmetric.cpp in order to reduce compile time memory
// usage per cpp file.

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

// \brief Test evaluation of rank 3 tensors with symmetry
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_rank_3() {
  // first and second indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, DataType, Symmetry<2, 2, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>, ti::b,
      ti::a, ti::C, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_3<
      false, DataType, Symmetry<2, 2, 1>,
      indextype_list<spatial_index, spatial_index, spacetime_index>, ti::L,
      ti::J, ti::I, Frame::Grid, Symmetry<3, 2, 1>>();

  // first and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, DataType, Symmetry<1, 2, 1>,
      indextype_list<spatial_index, spatial_index, spatial_index>, ti::i, ti::k,
      ti::j, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_3<
      false, DataType, Symmetry<1, 2, 1>,
      indextype_list<spacetime_index, spatial_index, spacetime_index>, ti::f,
      ti::M, ti::c, Frame::Grid, Symmetry<3, 2, 1>>();

  // second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, DataType, Symmetry<2, 1, 1>,
      indextype_list<spacetime_index, spatial_index, spatial_index>, ti::c,
      ti::I, ti::K, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_3<
      false, DataType, Symmetry<2, 1, 1>,
      indextype_list<spatial_index, spacetime_index, spacetime_index>, ti::J,
      ti::b, ti::c, Frame::Grid, Symmetry<3, 2, 1>>();

  // fully symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      true, DataType, Symmetry<1, 1, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>, ti::f,
      ti::d, ti::a, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_3<
      false, DataType, Symmetry<1, 1, 1>,
      indextype_list<spatial_index, spatial_index, spatial_index>, ti::k, ti::l,
      ti::i, Frame::Grid, Symmetry<2, 2, 1>>();
  TestHelpers::tenex::test_evaluate_rank_3<
      false, DataType, Symmetry<1, 1, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>, ti::g,
      ti::b, ti::c, Frame::Inertial, Symmetry<1, 2, 1>>();
  TestHelpers::tenex::test_evaluate_rank_3<
      false, DataType, Symmetry<1, 1, 1>,
      indextype_list<spatial_index, spatial_index, spatial_index>, ti::M, ti::N,
      ti::J, Frame::Grid, Symmetry<2, 1, 1>>();
  TestHelpers::tenex::test_evaluate_rank_3<
      false, DataType, Symmetry<1, 1, 1>,
      indextype_list<spacetime_index, spacetime_index, spacetime_index>, ti::E,
      ti::A, ti::B, Frame::Distorted, Symmetry<3, 2, 1>>();
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3Symmetric",
    "[DataStructures][Unit]") {
  test_evaluate_rank_3<double>();
  test_evaluate_rank_3<DataVector>();
}
