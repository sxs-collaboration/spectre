// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"

namespace {
// \brief Test evaluation of rank 4 tensors
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_rank_4() {
  // nonsymmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataType, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::b, ti::A, ti::k, ti::l>();

  // second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataType, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>,
      ti::G, ti::d, ti::a, ti::j>();

  // first, second, and fourth indices symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataType, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::i, ti::k, ti::l>();

  // fully symmetric
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataType, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::F, ti::A, ti::C, ti::D>();

  // different LHS symmetry
  TestHelpers::tenex::test_evaluate_rank_4<
      false, DataType, Symmetry<2, 1, 2, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::i, ti::k, ti::l, Symmetry<3, 2, 1, 1>>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank4",
                  "[DataStructures][Unit]") {
  test_evaluate_rank_4<double>();
  test_evaluate_rank_4<DataVector>();
}
