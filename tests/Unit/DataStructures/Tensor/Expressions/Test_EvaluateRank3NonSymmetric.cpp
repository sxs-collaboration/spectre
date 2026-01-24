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

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3NonSymmetric",
    "[DataStructures][Unit]") {
  // Rank 3: double; nonsymmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      double, Symmetry<3, 2, 1>, SpacetimeIndex, SpatialIndex, SpacetimeIndex,
      ti::D, ti::j, ti::B, Frame::Inertial>();

  // Rank 3: DataVector; nonsymmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      DataVector, Symmetry<3, 2, 1>, SpacetimeIndex, SpatialIndex,
      SpacetimeIndex, ti::D, ti::j, ti::B, Frame::Grid>();
}
