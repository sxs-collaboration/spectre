// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank0TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank1TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank3TestHelpers.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Evaluate",
                  "[DataStructures][Unit]") {
  // Rank 0: double
  TestHelpers::TensorExpressions::test_evaluate_rank_0<double>(-7.31);

  // Rank 0: DataVector
  TestHelpers::TensorExpressions::test_evaluate_rank_0<DataVector>(
      DataVector{-3.1, 9.4, 0.0, -3.1, 2.4, 9.8});

  // Rank 1: double; spacetime
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpacetimeIndex,
                                                       UpLo::Lo>(ti_a);
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpacetimeIndex,
                                                       UpLo::Lo>(ti_b);
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpacetimeIndex,
                                                       UpLo::Up>(ti_A);
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpacetimeIndex,
                                                       UpLo::Up>(ti_B);

  // Rank 1: double; spatial
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpatialIndex,
                                                       UpLo::Lo>(ti_i);
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpatialIndex,
                                                       UpLo::Lo>(ti_j);
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpatialIndex,
                                                       UpLo::Up>(ti_I);
  TestHelpers::TensorExpressions::test_evaluate_rank_1<double, SpatialIndex,
                                                       UpLo::Up>(ti_J);

  // Rank 1: DataVector
  TestHelpers::TensorExpressions::test_evaluate_rank_1<DataVector, SpatialIndex,
                                                       UpLo::Up>(ti_L);

  // Rank 2: double; nonsymmetric; spacetime only
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_a, ti_b);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(ti_A, ti_B);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_d, ti_c);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(ti_D, ti_C);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(ti_e, ti_F);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo>(ti_F, ti_e);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(ti_g, ti_B);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo>(ti_G, ti_b);

  // Rank 2: double; nonsymmetric; spatial only
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(ti_i, ti_j);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(ti_I, ti_J);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(ti_j, ti_i);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Up>(ti_J, ti_I);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(ti_i, ti_J);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(ti_I, ti_j);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(ti_j, ti_I);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(ti_J, ti_i);

  // Rank 2: double; nonsymmetric; spacetime and spatial mixed
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(ti_c, ti_I);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Lo>(ti_A, ti_i);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo>(ti_J, ti_a);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(ti_i, ti_B);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Lo>(ti_e, ti_j);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_i, ti_d);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Up, UpLo::Up>(ti_C, ti_I);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Up>(ti_J, ti_A);

  // Rank 2: double; symmetric; spacetime
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      double, SpacetimeIndex, UpLo::Lo>(ti_a, ti_d);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      double, SpacetimeIndex, UpLo::Up>(ti_G, ti_B);

  // Rank 2: double; symmetric; spatial
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      double, SpatialIndex, UpLo::Lo>(ti_j, ti_i);
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      double, SpatialIndex, UpLo::Up>(ti_I, ti_J);

  // Rank 2: DataVector; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_2_no_symmetry<
      DataVector, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(ti_f,
                                                                      ti_G);

  // Rank 2: DataVector; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_2_symmetric<
      DataVector, SpatialIndex, UpLo::Lo>(ti_j, ti_i);

  // Rank 3: double; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_no_symmetry<
      double, SpacetimeIndex, SpatialIndex, SpacetimeIndex, UpLo::Up, UpLo::Lo,
      UpLo::Up>(ti_D, ti_j, ti_B);

  // Rank 3: double; first and second indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_ab_symmetry<
      double, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(ti_b, ti_a,
                                                                  ti_C);

  // Rank 3: double; first and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_ac_symmetry<
      double, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_i, ti_f,
                                                                ti_j);

  // Rank 3: double; second and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_bc_symmetry<
      double, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(ti_d, ti_J,
                                                                ti_I);

  // Rank 3: double; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_abc_symmetry<
      double, SpacetimeIndex, UpLo::Lo>(ti_f, ti_d, ti_a);

  // Rank 3: DataVector; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_no_symmetry<
      DataVector, SpacetimeIndex, SpatialIndex, SpacetimeIndex, UpLo::Up,
      UpLo::Lo, UpLo::Up>(ti_D, ti_j, ti_B);

  // Rank 3: DataVector; first and second indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_ab_symmetry<
      DataVector, SpacetimeIndex, SpacetimeIndex, UpLo::Lo, UpLo::Up>(
      ti_b, ti_a, ti_C);

  // Rank 3: DataVector; first and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_ac_symmetry<
      DataVector, SpatialIndex, SpacetimeIndex, UpLo::Lo, UpLo::Lo>(ti_i, ti_f,
                                                                    ti_j);

  // Rank 3: DataVector; second and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_bc_symmetry<
      DataVector, SpacetimeIndex, SpatialIndex, UpLo::Lo, UpLo::Up>(ti_d, ti_J,
                                                                    ti_I);

  // Rank 3: DataVector; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_3_abc_symmetry<
      DataVector, SpacetimeIndex, UpLo::Lo>(ti_f, ti_d, ti_a);

  // Rank 4: double; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      double, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>(ti_b, ti_A, ti_k,
                                                              ti_l);

  // Rank 4: double; second and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      double, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>>(ti_G, ti_d, ti_a,
                                                          ti_j);

  // Rank 4: double; first, second, and fourth indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      double, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>(ti_j, ti_i, ti_k,
                                                              ti_l);

  // Rank 4: double; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      double, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>(ti_F, ti_A, ti_C,
                                                            ti_D);

  // Rank 4: DataVector; nonsymmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      DataVector, Symmetry<4, 3, 2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<1, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>>(ti_b, ti_A, ti_k,
                                                              ti_l);

  // Rank 4: DataVector; second and third indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      DataVector, Symmetry<3, 2, 2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                 SpatialIndex<1, UpLo::Lo, Frame::Grid>>>(ti_G, ti_d, ti_a,
                                                          ti_j);

  // Rank 4: DataVector; first, second, and fourth indices symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      DataVector, Symmetry<2, 2, 1, 2>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>(ti_j, ti_i, ti_k,
                                                              ti_l);

  // Rank 4: DataVector; symmetric
  TestHelpers::TensorExpressions::test_evaluate_rank_4<
      DataVector, Symmetry<1, 1, 1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>(ti_F, ti_A, ti_C,
                                                            ti_D);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubtract",
                  "[DataStructures][Unit]") {
  // Test adding scalars
  const Tensor<double> scalar_1{{{2.1}}};
  const Tensor<double> scalar_2{{{-0.8}}};
  Tensor<double> lhs_scalar =
      TensorExpressions::evaluate(scalar_1() + scalar_2());
  CHECK(lhs_scalar.get() == 1.3);

  Tensor<double, Symmetry<1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Hll{};
  Tensor<double, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      HHll{};
  std::iota(Hll.begin(), Hll.end(), 0.0);
  std::iota(HHll.rbegin(), HHll.rend(), -Hll.size());
  /// [use_tensor_index]
  auto Gll = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b) +
                                                         Hll(ti_a, ti_b));
  auto Gll2 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(All(ti_a, ti_b) +
                                                          Hll(ti_b, ti_a));
  auto Gll3 = TensorExpressions::evaluate<ti_a_t, ti_b_t>(
      All(ti_a, ti_b) + Hll(ti_b, ti_a) + All(ti_b, ti_a) - Hll(ti_b, ti_a));
  /// [use_tensor_index]
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      CHECK(Gll.get(i, j) == All.get(i, j) + Hll.get(i, j));
      CHECK(Gll2.get(i, j) == All.get(i, j) + Hll.get(j, i));
      CHECK(Gll3.get(i, j) == 2.0 * All.get(i, j));
    }
  }
  // Test 3 indices add subtract
  Tensor<double, Symmetry<1, 1, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Alll{};
  std::iota(Alll.begin(), Alll.end(), 0.0);
  Tensor<double, Symmetry<1, 2, 3>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Hlll{};
  Tensor<double, Symmetry<1, 2, 3>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      HHlll{};
  std::iota(Hlll.begin(), Hlll.end(), 0.0);
  std::iota(HHlll.rbegin(), HHlll.rend(), -Hlll.size());
  auto Glll = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_a, ti_b, ti_c));
  auto Glll2 = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c));
  auto Glll3 = TensorExpressions::evaluate<ti_a_t, ti_b_t, ti_c_t>(
      Alll(ti_a, ti_b, ti_c) + Hlll(ti_b, ti_a, ti_c) + Alll(ti_b, ti_a, ti_c) -
      Hlll(ti_b, ti_a, ti_c));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        CHECK(Glll.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(i, j, k));
        CHECK(Glll2.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(j, i, k));
        CHECK(Glll3.get(i, j, k) == 2.0 * Alll.get(i, j, k));
      }
    }
  }
}
