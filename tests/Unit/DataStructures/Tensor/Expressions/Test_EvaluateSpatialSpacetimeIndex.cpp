// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"

namespace {
// \brief Test evaluation of tensors where generic spatial indices are used for
// RHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs() {
  // test RHS Symmetry<2, 1>
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::i, ti::a, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::A, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::a, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::I, ti::a, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::i, ti::A, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::A, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::I, ti::A, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>,
      ti::J, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>,
      ti::J, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Grid>>,
      ti::J, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::l, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::l, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::l, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::K, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>,
      ti::K, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::K, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>>();

  // test RHS Symmetry<1, 1> to LHS Symmetry<1, 1> and <2, 1>
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::a, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::A, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::I, ti::A, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Distorted>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>,
      ti::J, ti::I, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Distorted>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Distorted>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Distorted>>,
      ti::J, ti::I, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Distorted>,
                 SpatialIndex<3, UpLo::Up, Frame::Distorted>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      true, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::j, ti::k, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::j, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
}

// \brief Test evaluation of tensors where generic spatial indices are used for
// LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_lhs() {
  // test RHS Symmetry<2, 1>
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::A, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::a, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpatialIndex<3, UpLo::Up, Frame::Grid>>,
      ti::A, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::a, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::I, ti::a, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::i, ti::A, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti::I, ti::A, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::J, ti::k, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::J, ti::k, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::J, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::l, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::K, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::K, ti::I, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::K, ti::I, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>();

  // test RHS Symmetry<1, 1> to LHS Symmetry<1, 1> and <2, 1>
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::K, ti::J, Symmetry<1, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::K, ti::J, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::K, ti::J, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::K, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::l, ti::k, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::l, ti::k, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::l, ti::k, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::l, ti::k, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>();
}

// \brief Test evaluation of rank 2 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs_and_lhs_rank2() {
  // test RHS Symmetry<2, 1>

  // - two lower spatial tensor indices
  // - RHS one or two spacetime indices
  // - LHS two spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::n, ti::m, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::m, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - upper spatial tensor index, lower spatial tensor index
  // - RHS two spacetime indices
  // - LHS one or two spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::I, ti::j, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::K, ti::i, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::N, ti::k, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();

  // - lower spatial tensor indices
  // - RHS spatial index, spacetime index
  // - LHS spatial index, spacetime index
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::j, ti::k>();

  // - upper spatial tensor indices
  // - RHS spacetime index, spatial index
  // - LHS spatial index, spacetime index
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::L, ti::J, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>();

  // - lower spatial tensor index, upper spatial tensor index
  // - RHS spatial index, spacetime index
  // - LHS spacetime index, spatial index
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::i, ti::M, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  // - lower spatial tensor indices
  // - RHS spacetime index, spatial index
  // - LHS spatial index, spacetime index
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::k, ti::j>();

  // - upper spatial tensor indices
  // - RHS spatial index, spacetime index
  // - LHS spacetime index, spatial index
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::L, ti::J, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Up, Frame::Inertial>>>();

  // - lower spacetime tensor index, lower spatial tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i>();

  // - lower spatial tensor index, lower spacetime tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::a>();

  // - upper spacetime tensor index, upper spatial tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::A, ti::I>();

  // - upper spatial tensor index, upper spacetime tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::I, ti::A>();

  // - lower spatial tensor index, upper spacetime tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::i, ti::A>();

  // - lower spacetime tensor index, upper spatial tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::a, ti::I>();

  // - upper spatial tensor index, lower spacetime tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::I, ti::a>();

  // - upper spacetime tensor index, lower spatial tensor index
  // - RHS spacetime indices
  // - LHS spacetime indices
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::A, ti::i>();

  // test RHS Symmetry<1, 1> to LHS Symmetry<1, 1> and <2, 1>

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::i, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::A, ti::I>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::A, ti::I, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::a>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::a, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::I, ti::A>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::I, ti::A, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::j, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::J, ti::I>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::J, ti::I, Symmetry<2, 1>>();
}

// \brief Test evaluation of rank 4 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
template <typename DataType>
void test_rhs_and_lhs_rank4() {
  // tests that return type is what is expected
  TestHelpers::tenex::test_evaluate_rank_4<
      true, DataType, Symmetry<1, 2, 2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
      ti::i, ti::k, ti::a, ti::j, Symmetry<1, 3, 2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Grid>>>();

  // tests that only spatial components are filled for LHS tensor arg
  TestHelpers::tenex::test_evaluate_rank_4<
      false, DataType, Symmetry<1, 1, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::m, ti::c, ti::i, ti::j, Symmetry<1, 1, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>();
}

// \brief Test evaluation of tensors where generic spatial indices are used for
// spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_spatial_spacetime_index() {
  test_rhs<DataType>();
  test_lhs<DataType>();
  test_rhs_and_lhs_rank2<DataType>();
  test_rhs_and_lhs_rank4<DataType>();
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateSpatialSpacetimeIndex",
    "[DataStructures][Unit]") {
  test_evaluate_spatial_spacetime_index<double>();
  test_evaluate_spatial_spacetime_index<DataVector>();
}
