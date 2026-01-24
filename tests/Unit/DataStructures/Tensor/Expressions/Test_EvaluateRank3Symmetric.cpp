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

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateRank3Symmetric",
    "[DataStructures][Unit]") {
  // Rank 3: double; first and second indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<
      double, Symmetry<2, 2, 1>, SpacetimeIndex, SpacetimeIndex, ti::b, ti::a,
      ti::C, Frame::Inertial>();

  // Rank 3: double; first and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<double, Symmetry<1, 2, 1>,
                                           SpatialIndex, SpacetimeIndex, ti::i,
                                           ti::f, ti::j, Frame::Grid>();

  // Rank 3: double; second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<double, Symmetry<2, 1, 1>,
                                           SpacetimeIndex, SpatialIndex, ti::d,
                                           ti::J, ti::I, Frame::Distorted>();

  // Rank 3: double; symmetric
  TestHelpers::tenex::test_evaluate_rank_3<double, Symmetry<1, 1, 1>,
                                           SpacetimeIndex, ti::f, ti::d, ti::a,
                                           Frame::Inertial>();

  // Rank 3: DataVector; first and second indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<DataVector, Symmetry<2, 2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::b, ti::a, ti::C, Frame::Grid>();

  // Rank 3: DataVector; first and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<DataVector, Symmetry<1, 2, 1>,
                                           SpatialIndex, SpacetimeIndex, ti::i,
                                           ti::f, ti::j, Frame::Distorted>();

  // Rank 3: DataVector; second and third indices symmetric
  TestHelpers::tenex::test_evaluate_rank_3<DataVector, Symmetry<2, 1, 1>,
                                           SpacetimeIndex, SpatialIndex, ti::d,
                                           ti::J, ti::I, Frame::Inertial>();

  // Rank 3: DataVector; symmetric
  TestHelpers::tenex::test_evaluate_rank_3<DataVector, Symmetry<1, 1, 1>,
                                           SpacetimeIndex, ti::f, ti::d, ti::a,
                                           Frame::Grid>();
}
