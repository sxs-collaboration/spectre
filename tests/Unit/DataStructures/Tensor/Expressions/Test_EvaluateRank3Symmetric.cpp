// Distributed under the MIT License.
// See LICENSE.txt for details.

// Rank 3 test cases for tenex::evaluate are split into this file and
// Test_EvaluateRank3NonSymmetric.cpp in order to reduce compile time memory
// usage per cpp file.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3Symmetric",
    "[DataStructures][Unit]") {
  // Rank 3: double; first and second indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3_ab_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up, ti::b, ti::a,
      ti::C>();

  // Rank 3: double; first and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3_ac_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo, ti::i, ti::f,
      ti::j>();

  // Rank 3: double; second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3_bc_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up, ti::d, ti::J,
      ti::I>();

  // Rank 3: double; symmetric
  TestHelpers::tenex::test_evaluate_rank_3_abc_symmetry<
      double, SpacetimeIndex, UpLo::Lo, ti::f, ti::d, ti::a>();

  // Rank 3: DataVector; first and second indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3_ab_symmetry<
      DataVector, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up, ti::b,
      ti::a, ti::C>();

  // Rank 3: DataVector; first and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3_ac_symmetry<
      DataVector, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo, ti::i,
      ti::f, ti::j>();

  // Rank 3: DataVector; second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3_bc_symmetry<
      DataVector, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up, ti::d,
      ti::J, ti::I>();

  // Rank 3: DataVector; symmetric
  TestHelpers::tenex::test_evaluate_rank_3_abc_symmetry<
      DataVector, SpacetimeIndex, UpLo::Lo, ti::f, ti::d, ti::a>();
}
