// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank0.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank1.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"

namespace {
template <auto&... TensorIndices>
void test_contains_indices_to_contract_impl(const bool expected) {
  CHECK(tenex::detail::contains_indices_to_contract<sizeof...(TensorIndices)>(
            {{std::decay_t<decltype(TensorIndices)>::value...}}) == expected);
}

void test_contains_indices_to_contract() {
  test_contains_indices_to_contract_impl<ti::a, ti::b, ti::c>(false);
  test_contains_indices_to_contract_impl<ti::I, ti::j>(false);
  test_contains_indices_to_contract_impl<ti::j>(false);
  test_contains_indices_to_contract_impl(false);

  test_contains_indices_to_contract_impl<ti::d, ti::D>(true);
  test_contains_indices_to_contract_impl<ti::I, ti::i>(true);
  test_contains_indices_to_contract_impl<ti::a, ti::K, ti::B, ti::b>(true);
  test_contains_indices_to_contract_impl<ti::j, ti::c, ti::J, ti::A, ti::a>(
      true);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Evaluate",
                  "[DataStructures][Unit]") {
  test_contains_indices_to_contract();

  // Rank 0: double
  TestHelpers::tenex::test_evaluate_rank_0<double>(-7.31);

  // Rank 0: DataVector
  TestHelpers::tenex::test_evaluate_rank_0<DataVector>(
      DataVector{-3.1, 9.4, 0.0, -3.1, 2.4, 9.8});

  // Rank 1: double; spacetime
  TestHelpers::tenex::test_evaluate_rank_1<double, SpacetimeIndex, ti::a>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpacetimeIndex, ti::b>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpacetimeIndex, ti::A>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpacetimeIndex, ti::B>();

  // Rank 1: double; spatial
  TestHelpers::tenex::test_evaluate_rank_1<double, SpatialIndex, ti::i>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpatialIndex, ti::j>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpatialIndex, ti::I>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpatialIndex, ti::J>();

  // Rank 1: DataVector
  TestHelpers::tenex::test_evaluate_rank_1<DataVector, SpatialIndex, ti::L>();

  // Rank 2: double; nonsymmetric; spacetime only
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, ti::a, ti::b>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, ti::A, ti::B>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, ti::d, ti::c>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, ti::D, ti::C>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, ti::e, ti::F>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, ti::F, ti::e>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, ti::g, ti::B>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, ti::G, ti::b>();

  // Rank 2: double; nonsymmetric; spatial only
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, ti::i, ti::j>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, ti::I, ti::J>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, ti::j, ti::i>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, ti::J, ti::I>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, ti::i, ti::J>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, ti::I, ti::j>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, ti::j, ti::I>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, ti::J, ti::i>();

  // Rank 2: double; nonsymmetric; spacetime and spatial mixed
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, ti::c, ti::I>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, ti::A, ti::i>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, ti::J, ti::a>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, ti::i, ti::B>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, ti::e, ti::j>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, ti::i, ti::d>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, ti::C, ti::I>();
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, ti::J, ti::A>();

  // Rank 2: double; symmetric; spacetime
  TestHelpers::tenex::test_evaluate_rank_2_symmetric<double, SpacetimeIndex,
                                                     ti::a, ti::d>();
  TestHelpers::tenex::test_evaluate_rank_2_symmetric<double, SpacetimeIndex,
                                                     ti::G, ti::B>();

  // Rank 2: double; symmetric; spatial
  TestHelpers::tenex::test_evaluate_rank_2_symmetric<double, SpatialIndex,
                                                     ti::j, ti::i>();
  TestHelpers::tenex::test_evaluate_rank_2_symmetric<double, SpatialIndex,
                                                     ti::I, ti::J>();

  // Rank 2: DataVector; nonsymmetric
  TestHelpers::tenex::test_evaluate_rank_2_no_symmetry<
      DataVector, SpacetimeIndex, SpacetimeIndex, ti::f, ti::G>();

  // Rank 2: DataVector; symmetric
  TestHelpers::tenex::test_evaluate_rank_2_symmetric<DataVector, SpatialIndex,
                                                     ti::j, ti::i>();
}
