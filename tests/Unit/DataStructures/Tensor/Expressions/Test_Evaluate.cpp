// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
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
  TestHelpers::tenex::test_evaluate_rank_1<double, SpacetimeIndex, ti::a,
                                           Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpacetimeIndex, ti::b,
                                           Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpacetimeIndex, ti::A,
                                           Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpacetimeIndex, ti::B,
                                           Frame::Inertial>();

  // Rank 1: double; spatial
  TestHelpers::tenex::test_evaluate_rank_1<double, SpatialIndex, ti::i,
                                           Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpatialIndex, ti::j,
                                           Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpatialIndex, ti::I,
                                           Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_1<double, SpatialIndex, ti::J,
                                           Frame::Grid>();

  // Rank 1: DataVector
  TestHelpers::tenex::test_evaluate_rank_1<DataVector, SpatialIndex, ti::L,
                                           Frame::Inertial>();

  // Rank 2: double; nonsymmetric; spacetime only
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::a, ti::b, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::A, ti::B, Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::d, ti::c, Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::D, ti::C, Frame::NoFrame>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::e, ti::F, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::F, ti::e, Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::g, ti::B, Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::G, ti::b, Frame::NoFrame>();

  // Rank 2: double; nonsymmetric; spatial only
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpatialIndex, ti::i, ti::j,
                                           Frame::NoFrame>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpatialIndex, ti::I, ti::J,
                                           Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpatialIndex, ti::j, ti::i,
                                           Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpatialIndex, ti::J, ti::I,
                                           Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpatialIndex, ti::i, ti::J,
                                           Frame::NoFrame>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpatialIndex, ti::I, ti::j,
                                           Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpatialIndex, ti::j, ti::I,
                                           Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpatialIndex, ti::J, ti::i,
                                           Frame::Inertial>();

  // Rank 2: double; nonsymmetric; spacetime and spatial mixed
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpatialIndex, ti::c,
                                           ti::I, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpatialIndex, ti::A,
                                           ti::i, Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpacetimeIndex, ti::J, ti::a,
                                           Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpacetimeIndex, ti::i, ti::B,
                                           Frame::NoFrame>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpatialIndex, ti::e,
                                           ti::j, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpacetimeIndex, ti::i, ti::d,
                                           Frame::Grid>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>,
                                           SpacetimeIndex, SpatialIndex, ti::C,
                                           ti::I, Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<2, 1>, SpatialIndex,
                                           SpacetimeIndex, ti::J, ti::A,
                                           Frame::NoFrame>();

  // Rank 2: double; symmetric; spacetime
  TestHelpers::tenex::test_evaluate_rank_2<
      double, Symmetry<1, 1>, SpacetimeIndex, ti::a, ti::d, Frame::Inertial>();
  TestHelpers::tenex::test_evaluate_rank_2<
      double, Symmetry<1, 1>, SpacetimeIndex, ti::G, ti::B, Frame::Grid>();

  // Rank 2: double; symmetric; spatial
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<1, 1>, SpatialIndex,
                                           ti::j, ti::i, Frame::Distorted>();
  TestHelpers::tenex::test_evaluate_rank_2<double, Symmetry<1, 1>, SpatialIndex,
                                           ti::I, ti::J, Frame::NoFrame>();

  // Rank 2: DataVector; nonsymmetric
  TestHelpers::tenex::test_evaluate_rank_2<DataVector, Symmetry<2, 1>,
                                           SpacetimeIndex, SpacetimeIndex,
                                           ti::f, ti::G, Frame::Inertial>();

  // Rank 2: DataVector; symmetric
  TestHelpers::tenex::test_evaluate_rank_2<
      DataVector, Symmetry<1, 1>, SpatialIndex, ti::j, ti::i, Frame::Grid>();
}
