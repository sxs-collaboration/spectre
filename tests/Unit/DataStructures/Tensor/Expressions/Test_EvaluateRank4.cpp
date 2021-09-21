// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank4",
                  "[DataStructures][Unit]") {
  // Rank 4: double; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      double, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti_b, ti_A, ti_k, ti_l>();

  // Rank 4: double; second and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      double, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>,
      ti_G, ti_d, ti_a, ti_j>();

  // Rank 4: double; first, second, and fourth indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      double, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti_j, ti_i, ti_k, ti_l>();

  // Rank 4: double; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      double, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti_F, ti_A, ti_C, ti_D>();

  // Rank 4: DataVector; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      DataVector, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti_b, ti_A, ti_k, ti_l>();

  // Rank 4: DataVector; second and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      DataVector, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>,
      ti_G, ti_d, ti_a, ti_j>();

  // Rank 4: DataVector; first, second, and fourth indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      DataVector, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti_j, ti_i, ti_k, ti_l>();

  // Rank 4: DataVector; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      DataVector, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>,
      ti_F, ti_A, ti_C, ti_D>();
}
