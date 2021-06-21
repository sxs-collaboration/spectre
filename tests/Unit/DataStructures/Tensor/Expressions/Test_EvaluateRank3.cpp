// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateRank3",
                  "[DataStructures][Unit]") {
  // Rank 3: double; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo,
      UpLo::Up, ti_D, ti_j, ti_B>();

  // Rank 3: double; first and second indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_ab_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up, ti_b, ti_a,
      ti_C>();

  // Rank 3: double; first and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_ac_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo, ti_i, ti_f,
      ti_j>();

  // Rank 3: double; second and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_bc_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up, ti_d, ti_J,
      ti_I>();

  // Rank 3: double; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_abc_symmetry<
      double, SpacetimeIndex, UpLo::Lo, ti_f, ti_d, ti_a>();

  // Rank 3: DataVector; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_no_symmetry<
      DataVector, SpacetimeIndex, SpatialIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Lo, UpLo::Up, ti_D, ti_j, ti_B>();

  // Rank 3: DataVector; first and second indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_ab_symmetry<
      DataVector, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up, ti_b,
      ti_a, ti_C>();

  // Rank 3: DataVector; first and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_ac_symmetry<
      DataVector, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo, ti_i, ti_f,
      ti_j>();

  // Rank 3: DataVector; second and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_bc_symmetry<
      DataVector, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up, ti_d, ti_J,
      ti_I>();

  // Rank 3: DataVector; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_abc_symmetry<
      DataVector, SpacetimeIndex, UpLo::Lo, ti_f, ti_d, ti_a>();
}
