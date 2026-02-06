// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank2.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/EvaluateRank4.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// \brief Test evaluation of tensors where concrete time indices are used for
// RHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename Generator, typename DataType>
void test_rhs(const gsl::not_null<Generator*> generator,
              const DataType& used_for_size) {
  // Note: this function doesn't utilize test helper functions like
  // test_evaluate() because they aren't generic enough to handle
  // test cases where the number of indices on the RHS and LHS are not equal.
  // Instead, we have to manually check each test case of interest.

  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;
  using frame = Frame::Inertial;

  // Rank 1 testing

  const auto R_a = make_with_random_values<Tensor<
      DataType, Symmetry<1>, index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_A = make_with_random_values<Tensor<
      DataType, Symmetry<1>, index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);

  // \f$L = R_{t}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Scalar<DataType> L_from_R_t = tenex::evaluate(R_a(ti::t));
  // \f$L = R_{T}\f$
  const Scalar<DataType> L_from_R_T = tenex::evaluate(R_A(ti::T));

  CHECK(get(L_from_R_t) == R_a.get(0));
  CHECK(get(L_from_R_T) == R_A.get(0));

  // Rank 2 testing

  // RHS tensors with non-symmetric spacetime indices
  const auto R_ab = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_AB = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_Ab = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_aB = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);
  // RHS tensors with one spacetime and one spatial index
  const auto R_ai = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                        SpatialIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_ia = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpatialIndex<dim, UpLo::Lo, frame>,
                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_IA = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpatialIndex<dim, UpLo::Up, frame>,
                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_AI = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                        SpatialIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_Ai = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                        SpatialIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_Ia = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpatialIndex<dim, UpLo::Up, frame>,
                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_aI = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                        SpatialIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_iA = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpatialIndex<dim, UpLo::Lo, frame>,
                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);
  // RHS tensors with symmetric spacetime indices
  const auto S_ab = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto S_AB = make_with_random_values<
      Tensor<DataType, Symmetry<1, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);

  // Evaluations of non-symmetric RHS tensors

  // \f$L_{a} = R_{at}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>
      L_a_from_R_at = tenex::evaluate<ti::a>(R_ab(ti::a, ti::t));
  // \f$L_{a} = R_{ta}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>
      L_a_from_R_ta = tenex::evaluate<ti::a>(R_ab(ti::t, ti::a));
  // \f$L = R_{tt}\f$
  const Scalar<DataType> L_from_R_tt = tenex::evaluate(R_ab(ti::t, ti::t));
  // \f$L^{a} = R^{at}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>
      L_A_from_R_AT = tenex::evaluate<ti::A>(R_AB(ti::A, ti::T));
  // \f$L^{a} = R^{ta}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>
      L_A_from_R_TA = tenex::evaluate<ti::A>(R_AB(ti::T, ti::A));
  // \f$L = R^{tt}\f$
  const Scalar<DataType> L_from_R_TT = tenex::evaluate(R_AB(ti::T, ti::T));
  // \f$L^{a} = R^{a}{}_{t}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>
      L_A_from_R_At = tenex::evaluate<ti::A>(R_Ab(ti::A, ti::t));
  // \f$L_{a} = R^{t}{}_{a}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>
      L_a_from_R_Ta = tenex::evaluate<ti::a>(R_Ab(ti::T, ti::a));
  // \f$L = R^{t}{}_{t}\f$
  const Scalar<DataType> L_from_R_Tt = tenex::evaluate(R_Ab(ti::T, ti::t));
  // \f$L_{a} = R_{a}{}^{t}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>
      L_a_from_R_aT = tenex::evaluate<ti::a>(R_aB(ti::a, ti::T));
  // \f$L^{a} = R_{t}{}^{a}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>
      L_A_from_R_tA = tenex::evaluate<ti::A>(R_aB(ti::t, ti::A));
  // \f$L = R_{t}{}^{t}\f$
  const Scalar<DataType> L_from_R_tT = tenex::evaluate(R_aB(ti::t, ti::T));

  // Evaluations of symmetric RHS tensors

  // \f$L_{a} = S_{at}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>
      L_a_from_S_at = tenex::evaluate<ti::a>(S_ab(ti::a, ti::t));
  // \f$L_{a} = S_{ta}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>
      L_a_from_S_ta = tenex::evaluate<ti::a>(S_ab(ti::t, ti::a));
  // \f$L = S_{tt}\f$
  const Scalar<DataType> L_from_S_tt = tenex::evaluate(S_ab(ti::t, ti::t));
  // \f$L^{a} = S^{at}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>
      L_A_from_S_AT = tenex::evaluate<ti::A>(S_AB(ti::A, ti::T));
  // \f$L^{a} = S^{ta}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>
      L_A_from_S_TA = tenex::evaluate<ti::A>(S_AB(ti::T, ti::A));
  // \f$L = S^{tt}\f$
  const Scalar<DataType> L_from_S_TT = tenex::evaluate(S_AB(ti::T, ti::T));

  // Evaluations of RHS tensors with one spatial and one spacetime index

  // \f$L_{i} = R_{it}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_R_it = tenex::evaluate<ti::i>(R_ia(ti::i, ti::t));
  // \f$L_{i} = R_{ti}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_R_ti = tenex::evaluate<ti::i>(R_ai(ti::t, ti::i));
  // \f$L^{i} = R^{it}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_R_IT = tenex::evaluate<ti::I>(R_IA(ti::I, ti::T));
  // \f$L^{i} = R^{ti}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_R_TI = tenex::evaluate<ti::I>(R_AI(ti::T, ti::I));
  // \f$L^{i} = R^{i}{}_{t}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_R_It = tenex::evaluate<ti::I>(R_Ia(ti::I, ti::t));
  // \f$L_{i} = R^{t}{}_{i}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_R_Ti = tenex::evaluate<ti::i>(R_Ai(ti::T, ti::i));
  // \f$L_{i} = R_{i}{}^{t}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Lo, frame>>>
      L_i_from_R_iT = tenex::evaluate<ti::i>(R_iA(ti::i, ti::T));
  // \f$L^{i} = R_{t}{}^{i}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<dim, UpLo::Up, frame>>>
      L_I_from_R_tI = tenex::evaluate<ti::I>(R_aI(ti::t, ti::I));

  CHECK(get(L_from_R_tt) == R_ab.get(0, 0));
  CHECK(get(L_from_R_TT) == R_AB.get(0, 0));
  CHECK(get(L_from_R_Tt) == R_Ab.get(0, 0));
  CHECK(get(L_from_R_tT) == R_aB.get(0, 0));

  CHECK(get(L_from_S_tt) == S_ab.get(0, 0));
  CHECK(get(L_from_S_TT) == S_AB.get(0, 0));

  for (size_t a = 0; a < dim + 1; a++) {
    CHECK(L_a_from_R_at.get(a) == R_ab.get(a, 0));
    CHECK(L_a_from_R_ta.get(a) == R_ab.get(0, a));
    CHECK(L_A_from_R_AT.get(a) == R_AB.get(a, 0));
    CHECK(L_A_from_R_TA.get(a) == R_AB.get(0, a));
    CHECK(L_A_from_R_At.get(a) == R_Ab.get(a, 0));
    CHECK(L_a_from_R_Ta.get(a) == R_Ab.get(0, a));
    CHECK(L_a_from_R_aT.get(a) == R_aB.get(a, 0));
    CHECK(L_A_from_R_tA.get(a) == R_aB.get(0, a));

    CHECK(L_a_from_S_at.get(a) == S_ab.get(a, 0));
    CHECK(L_a_from_S_ta.get(a) == S_ab.get(0, a));
    CHECK(L_A_from_S_AT.get(a) == S_AB.get(a, 0));
    CHECK(L_A_from_S_TA.get(a) == S_AB.get(0, a));
  }

  for (size_t i = 0; i < dim; i++) {
    CHECK(L_i_from_R_it.get(i) == R_ia.get(i, 0));
    CHECK(L_i_from_R_ti.get(i) == R_ai.get(0, i));
    CHECK(L_I_from_R_IT.get(i) == R_IA.get(i, 0));
    CHECK(L_I_from_R_TI.get(i) == R_AI.get(0, i));
    CHECK(L_I_from_R_It.get(i) == R_Ia.get(i, 0));
    CHECK(L_i_from_R_Ti.get(i) == R_Ai.get(0, i));
    CHECK(L_i_from_R_iT.get(i) == R_iA.get(i, 0));
    CHECK(L_I_from_R_tI.get(i) == R_aI.get(0, i));
  }
}

// \brief Test evaluation of tensors where concrete time indices are used for
// LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename Generator, typename DataType>
void test_lhs(const gsl::not_null<Generator*> generator,
              const DataType& used_for_size) {
  // Note: this function doesn't utilize test helper functions like
  // test_evaluate() because they aren't generic enough to handle
  // test cases where the number of indices on the RHS and LHS are not equal.
  // Instead, we have to manually check each test case of interest.

  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;
  using frame = Frame::Inertial;

  const auto R = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);

  // Test evaluation of RHS scalar to LHS rank 1

  // \f$L_{t} = R\f$
  //
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  //
  // Assign a placeholder to the LHS tensor's components before it is computed
  // so that when test expressions below only compute time components, we can
  // check that LHS spatial components haven't changed
  auto L_t_from_R =
      make_with_value<Tensor<DataType, Symmetry<1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t>(make_not_null(&L_t_from_R), R());
  // \f$L_{T} = R\f$
  auto L_T_from_R =
      make_with_value<Tensor<DataType, Symmetry<1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T>(make_not_null(&L_T_from_R), R());

  CHECK(L_t_from_R.get(0) == get(R));
  CHECK(L_T_from_R.get(0) == get(R));
  for (size_t i = 0; i < dim; i++) {
    CHECK(L_t_from_R.get(i + 1) ==
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
    CHECK(L_T_from_R.get(i + 1) ==
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  }

  // Test evaluation of RHS scalar to non-symmetric LHS rank 2

  // \f$L_{tt} = R\f$
  auto L_tt_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::t>(make_not_null(&L_tt_from_R), R());
  // \f$L^{tt} = R\f$
  auto L_TT_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::T>(make_not_null(&L_TT_from_R), R());
  // \f$L^{t}{}_{t} = R\f$
  auto L_Tt_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::t>(make_not_null(&L_Tt_from_R), R());
  // \f$L_{t}{}^{t} = R\f$
  auto L_tT_from_R =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::T>(make_not_null(&L_tT_from_R), R());

  // Test evaluation of RHS scalar to symmetric LHS rank 2

  // \f$M_{tt} = R\f$
  auto M_tt_from_R =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::t>(make_not_null(&M_tt_from_R), R());
  // \f$M^{tt} = R\f$
  auto M_TT_from_R =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::T>(make_not_null(&M_TT_from_R), R());
  // \f$M^{t}{}_{t} = R\f$

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = 0; b < dim + 1; b++) {
      if (a == 0 and b == 0) {
        CHECK(L_tt_from_R.get(0, 0) == get(R));
        CHECK(L_TT_from_R.get(0, 0) == get(R));
        CHECK(L_Tt_from_R.get(0, 0) == get(R));
        CHECK(L_tT_from_R.get(0, 0) == get(R));

        CHECK(M_tt_from_R.get(0, 0) == get(R));
        CHECK(M_TT_from_R.get(0, 0) == get(R));
      } else {
        CHECK(L_tt_from_R.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_TT_from_R.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_Tt_from_R.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(L_tT_from_R.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);

        CHECK(M_tt_from_R.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(M_TT_from_R.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
      }
    }
  }

  // Evaluations of non-symmetric LHS tensors

  const auto R_a = make_with_random_values<Tensor<
      DataType, Symmetry<1>, index_list<SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_A = make_with_random_values<Tensor<
      DataType, Symmetry<1>, index_list<SpacetimeIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);

  // \f$L_{at} = R_{a}\f$
  auto L_at_from_R_a =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::a, ti::t>(make_not_null(&L_at_from_R_a), R_a(ti::a));
  // \f$L_{ta} = R_{a}\f$
  auto L_ta_from_R_a =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::a>(make_not_null(&L_ta_from_R_a), R_a(ti::a));
  // \f$L^{at} = R^{a}\f$
  auto L_AT_from_R_A =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::A, ti::T>(make_not_null(&L_AT_from_R_A), R_A(ti::A));
  // \f$L^{ta} = R^{a}\f$
  auto L_TA_from_R_A =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::A>(make_not_null(&L_TA_from_R_A), R_A(ti::A));
  // \f$L^{a}{}_{t} = R^{a}\f$
  auto L_At_from_R_A =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::A, ti::t>(make_not_null(&L_At_from_R_A), R_A(ti::A));
  // \f$L^{t}{}_{a} = R_{a}\f$
  auto L_Ta_from_R_a =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::a>(make_not_null(&L_Ta_from_R_a), R_a(ti::a));
  // \f$L_{a}{}^{t} = R_{a}\f$
  auto L_aT_from_R_a =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::a, ti::T>(make_not_null(&L_aT_from_R_a), R_a(ti::a));
  // \f$L_{t}{}^{a} = R^{a}\f$
  auto L_tA_from_R_A =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::A>(make_not_null(&L_tA_from_R_A), R_A(ti::A));

  for (size_t a = 0; a < dim + 1; a++) {
    CHECK(L_at_from_R_a.get(a, 0) == R_a.get(a));
    CHECK(L_ta_from_R_a.get(0, a) == R_a.get(a));
    CHECK(L_AT_from_R_A.get(a, 0) == R_A.get(a));
    CHECK(L_TA_from_R_A.get(0, a) == R_A.get(a));
    CHECK(L_At_from_R_A.get(a, 0) == R_A.get(a));
    CHECK(L_Ta_from_R_a.get(0, a) == R_a.get(a));
    CHECK(L_aT_from_R_a.get(a, 0) == R_a.get(a));
    CHECK(L_tA_from_R_A.get(0, a) == R_A.get(a));

    for (size_t i = 0; i < dim; i++) {
      CHECK(L_at_from_R_a.get(a, i + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_ta_from_R_a.get(i + 1, a) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_AT_from_R_A.get(a, i + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_TA_from_R_A.get(i + 1, a) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_At_from_R_A.get(a, i + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_Ta_from_R_a.get(i + 1, a) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_aT_from_R_a.get(a, i + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_tA_from_R_A.get(i + 1, a) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
    }
  }

  // Evaluations of symmetric LHS tensors

  // \f$M_{at} = R_{a}\f$
  auto M_at_from_R_a =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::a, ti::t>(make_not_null(&M_at_from_R_a), R_a(ti::a));
  // \f$M_{ta} = R_{a}\f$
  auto M_ta_from_R_a =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::a>(make_not_null(&M_ta_from_R_a), R_a(ti::a));
  // \f$M^{at} = R^{a}\f$
  auto M_AT_from_R_A =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::A, ti::T>(make_not_null(&M_AT_from_R_A), R_A(ti::A));
  // \f$M^{ta} = R^{a}\f$
  auto M_TA_from_R_A =
      make_with_value<Tensor<DataType, Symmetry<1, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::A>(make_not_null(&M_TA_from_R_A), R_A(ti::A));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = a; b < dim + 1; b++) {
      if (a == 0) {
        CHECK(M_at_from_R_a.get(a, b) == R_a.get(b));
        CHECK(M_ta_from_R_a.get(a, b) == R_a.get(b));
        CHECK(M_AT_from_R_A.get(a, b) == R_A.get(b));
        CHECK(M_TA_from_R_A.get(a, b) == R_A.get(b));
      } else {
        CHECK(M_at_from_R_a.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(M_ta_from_R_a.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(M_AT_from_R_A.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        CHECK(M_TA_from_R_A.get(a, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
      }
    }
  }

  // Evaluations of RHS tensors with one spatial and one spacetime index

  const auto R_i = make_with_random_values<Tensor<
      DataType, Symmetry<1>, index_list<SpatialIndex<dim, UpLo::Lo, frame>>>>(
      generator, distribution, used_for_size);
  const auto R_I = make_with_random_values<Tensor<
      DataType, Symmetry<1>, index_list<SpatialIndex<dim, UpLo::Up, frame>>>>(
      generator, distribution, used_for_size);

  // \f$L_{it} = R_{i}\f$
  auto L_it_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpatialIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::i, ti::t>(make_not_null(&L_it_from_R_i), R_i(ti::i));
  // \f$L_{ti} = R_{i}\f$
  auto L_ti_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpatialIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::i>(make_not_null(&L_ti_from_R_i), R_i(ti::i));
  // \f$L^{it} = R^{i}\f$
  auto L_IT_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpatialIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::I, ti::T>(make_not_null(&L_IT_from_R_I), R_I(ti::I));
  // \f$L^{ti} = R^{i}\f$
  auto L_TI_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpatialIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::I>(make_not_null(&L_TI_from_R_I), R_I(ti::I));
  // \f$L^{i}{}_{t} = R^{i}\f$
  auto L_It_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpatialIndex<dim, UpLo::Up, frame>,
                                        SpacetimeIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::I, ti::t>(make_not_null(&L_It_from_R_I), R_I(ti::I));
  // \f$L^{t}{}_{i} = R_{i}\f$
  auto L_Ti_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Up, frame>,
                                        SpatialIndex<dim, UpLo::Lo, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::T, ti::i>(make_not_null(&L_Ti_from_R_i), R_i(ti::i));
  // \f$L_{i}{}^{t} = R_{i}\f$
  auto L_iT_from_R_i =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpatialIndex<dim, UpLo::Lo, frame>,
                                        SpacetimeIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::i, ti::T>(make_not_null(&L_iT_from_R_i), R_i(ti::i));
  // \f$L_{t}{}^{i} = R^{i}\f$
  auto L_tI_from_R_I =
      make_with_value<Tensor<DataType, Symmetry<2, 1>,
                             index_list<SpacetimeIndex<dim, UpLo::Lo, frame>,
                                        SpatialIndex<dim, UpLo::Up, frame>>>>(
          used_for_size,
          TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::t, ti::I>(make_not_null(&L_tI_from_R_I), R_I(ti::I));

  for (size_t i = 0; i < dim; i++) {
    CHECK(L_it_from_R_i.get(i, 0) == R_i.get(i));
    CHECK(L_ti_from_R_i.get(0, i) == R_i.get(i));
    CHECK(L_IT_from_R_I.get(i, 0) == R_I.get(i));
    CHECK(L_TI_from_R_I.get(0, i) == R_I.get(i));
    CHECK(L_It_from_R_I.get(i, 0) == R_I.get(i));
    CHECK(L_Ti_from_R_i.get(0, i) == R_i.get(i));
    CHECK(L_iT_from_R_i.get(i, 0) == R_i.get(i));
    CHECK(L_tI_from_R_I.get(0, i) == R_I.get(i));
    for (size_t j = 0; j < dim; j++) {
      CHECK(L_it_from_R_i.get(i, j + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_ti_from_R_i.get(j + 1, i) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_IT_from_R_I.get(i, j + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_TI_from_R_I.get(j + 1, i) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_It_from_R_I.get(i, j + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_Ti_from_R_i.get(j + 1, i) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_iT_from_R_i.get(i, j + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
      CHECK(L_tI_from_R_I.get(j + 1, i) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
    }
  }
}

// \brief Test evaluation of rank 2 tensors where concrete time indices are used
// for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_rhs_and_lhs_rank_2() {
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::t>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::a, ti::t, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::t, ti::a>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::t, ti::a, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::t, ti::t>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::t, ti::t, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::T, ti::T>();
  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<1, 1>,
      index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::T, ti::T, Symmetry<2, 1>>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::A, ti::t>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::T, ti::a>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::a, ti::T>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::t, ti::A>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::t, ti::i>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
      ti::i, ti::t>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::I, ti::t>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                 SpatialIndex<2, UpLo::Lo, Frame::Inertial>>,
      ti::T, ti::i>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                 SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
      ti::i, ti::T>();

  TestHelpers::tenex::test_evaluate_rank_2_impl<
      false, DataType, Symmetry<2, 1>,
      index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
      ti::t, ti::I>();
}

// \brief Test evaluation of rank 4 tensors where concrete time indices are used
// for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename Generator, typename DataType>
void test_rhs_and_lhs_rank_4(const gsl::not_null<Generator*> generator,
                             const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>>>>(
      generator, distribution, used_for_size);

  // \f$L_{a}{}^{t}{}_{tb} = R_{tba}\f$
  //
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  //
  // Assign a placeholder to the LHS tensor's components before it is computed
  // so that when test expressions below only compute time components, we can
  // check that LHS spatial components haven't changed
  auto LaTtb_from_R_tba = make_with_value<
      Tensor<DataType, Symmetry<2, 3, 2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Up, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>>>>(
      used_for_size,
      TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::a, ti::T, ti::t, ti::b>(make_not_null(&LaTtb_from_R_tba),
                                              R(ti::t, ti::b, ti::a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = 0; b < dim + 1; b++) {
      CHECK(LaTtb_from_R_tba.get(a, 0, 0, b) == R.get(0, b, a));

      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          CHECK(
              LaTtb_from_R_tba.get(a, i + 1, j + 1, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        }
      }
    }
  }
}

// \brief Test evaluation of tensors where concrete time indices are used for
// spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType>
void test_evaluate_time_index(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);

  test_rhs(make_not_null(&generator), used_for_size);
  test_lhs(make_not_null(&generator), used_for_size);
  test_rhs_and_lhs_rank_2<DataType>();
  test_rhs_and_lhs_rank_4(make_not_null(&generator), used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateTimeIndex",
                  "[DataStructures][Unit]") {
  test_evaluate_time_index(std::numeric_limits<double>::signaling_NaN());
  test_evaluate_time_index(
      DataVector(3, std::numeric_limits<double>::signaling_NaN()));
}
