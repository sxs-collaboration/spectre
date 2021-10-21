// Distributed under the MIT License.
// See LICENSE.txt for details.

// Rank 3 test cases for TensorExpressions::evaluate are split into this file
// and Test_EvaluateRank3Symmetric.cpp in order to reduce compile time memory
// usage per cpp file.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3.hpp"

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3NonSymmetric",
    "[DataStructures][Unit]") {
  // Rank 3: double; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo,
      UpLo::Up, ti_D, ti_j, ti_B>();

  // Rank 3: DataVector; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_no_symmetry<
      DataVector, SpacetimeIndex, SpatialIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Lo, UpLo::Up, ti_D, ti_j, ti_B>();
}
