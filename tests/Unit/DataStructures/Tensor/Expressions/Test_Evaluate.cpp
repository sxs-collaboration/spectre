// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank0TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank1TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2TestHelpers.hpp"

namespace {
template <auto&... TensorIndices>
void test_contains_indices_to_contract_impl(const bool expected) {
  CHECK(TensorExpressions::detail::contains_indices_to_contract<sizeof...(
            TensorIndices)>(
            {{std::decay_t<decltype(TensorIndices)>::value...}}) == expected);
}

void test_contains_indices_to_contract() {
  test_contains_indices_to_contract_impl<ti_a, ti_b, ti_c>(false);
  test_contains_indices_to_contract_impl<ti_I, ti_j>(false);
  test_contains_indices_to_contract_impl<ti_j>(false);
  test_contains_indices_to_contract_impl(false);

  test_contains_indices_to_contract_impl<ti_d, ti_D>(true);
  test_contains_indices_to_contract_impl<ti_I, ti_i>(true);
  test_contains_indices_to_contract_impl<ti_a, ti_K, ti_B, ti_b>(true);
  test_contains_indices_to_contract_impl<ti_j, ti_c, ti_J, ti_A, ti_a>(true);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Evaluate",
                  "[DataStructures][Unit]") {
  test_contains_indices_to_contract();

  // Rank 0: double
  TestHelpers::TensorExpressions::test_evaluate_rank_0<double>(-7.31);

  // Rank 0: DataVector
  TestHelpers::TensorExpressions::test_evaluate_rank_0<DataVector>(
      DataVector{-3.1, 9.4, 0.0, -3.1, 2.4, 9.8});

  // Rank 1: double; spacetime
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpacetimeIndex,
                                                       UpLo::Lo, ti_a>();
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpacetimeIndex,
                                                       UpLo::Lo, ti_b>();
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpacetimeIndex,
                                                       UpLo::Up, ti_A>();
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpacetimeIndex,
                                                       UpLo::Up, ti_B>();

  // Rank 1: double; spatial
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpatialIndex,
                                                       UpLo::Lo, ti_i>();
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpatialIndex,
                                                       UpLo::Lo, ti_j>();
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpatialIndex,
                                                       UpLo::Up, ti_I>();
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpatialIndex,
                                                       UpLo::Up, ti_J>();

  // Rank 1: DataVector
  TestHelpers::TensorExpressions::test_evaluate_rank_1<DataVector, SpatialIndex,
                                                       UpLo::Up, ti_L>();

  // Rank 2: double; nonsymmetric; spacetime only
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo, ti_a, ti_b>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Up, ti_A, ti_B>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo, ti_d, ti_c>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Up, ti_D, ti_C>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up, ti_e, ti_F>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo, ti_F, ti_e>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up, ti_g, ti_B>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo, ti_G, ti_b>();

  // Rank 2: double; nonsymmetric; spatial only
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo, ti_i, ti_j>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up, ti_I, ti_J>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo, ti_j, ti_i>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up, ti_J, ti_I>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up, ti_i, ti_J>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo, ti_I, ti_j>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up, ti_j, ti_I>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo, ti_J, ti_i>();

  // Rank 2: double; nonsymmetric; spacetime and spatial mixed
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up, ti_c, ti_I>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Lo, ti_A, ti_i>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo, ti_J, ti_a>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up, ti_i, ti_B>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Lo, ti_e, ti_j>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo, ti_i, ti_d>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Up, ti_C, ti_I>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Up, ti_J, ti_A>();

  // Rank 2: double; symmetric; spacetime
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      double, SpacetimeIndex, UpLo::Lo, ti_a, ti_d>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      double, SpacetimeIndex, UpLo::Up, ti_G, ti_B>();

  // Rank 2: double; symmetric; spatial
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      double, SpatialIndex, UpLo::Lo, ti_j, ti_i>();
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      double, SpatialIndex, UpLo::Up, ti_I, ti_J>();

  // Rank 2: DataVector; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      DataVector, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up, ti_f,
      ti_G>();

  // Rank 2: DataVector; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      DataVector, SpatialIndex, UpLo::Lo, ti_j, ti_i>();
}
