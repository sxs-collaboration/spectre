// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// \brief Test evaluation of tensors where generic spatial indices are used for
// RHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType, typename Generator>
void test_rhs(const DataType& used_for_size,
              const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ai = TensorExpressions::evaluate<ti_a, ti_i>(R(ti_a, ti_i));

  // \f$L_{ia} = R_{ai}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ai = TensorExpressions::evaluate<ti_i, ti_a>(R(ti_a, ti_i));

  // \f$L_{ai} = R_{ia}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ia = TensorExpressions::evaluate<ti_a, ti_i>(R(ti_i, ti_a));

  // \f$L_{ia} = R_{ia}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ia = TensorExpressions::evaluate<ti_i, ti_a>(R(ti_i, ti_a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t i = 0; i < dim; i++) {
      CHECK(Lai_from_R_ai.get(a, i) == R.get(a, i + 1));
      CHECK(Lia_from_R_ai.get(i, a) == R.get(a, i + 1));
      CHECK(Lai_from_R_ia.get(a, i) == R.get(i + 1, a));
      CHECK(Lia_from_R_ia.get(i, a) == R.get(i + 1, a));
    }
  }
}

// \brief Test evaluation of tensors where generic spatial indices are used for
// LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType, typename Generator>
void test_lhs(const DataType& used_for_size,
              const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ai(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_i>(make_not_null(&Lai_from_R_ai),
                                          R(ti_a, ti_i));

  // \f$L_{ia} = R_{ai}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ai(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_a>(make_not_null(&Lia_from_R_ai),
                                          R(ti_a, ti_i));

  const auto S = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // \f$L_{ia} = S_{ia}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_S_ia(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_a>(make_not_null(&Lia_from_S_ia),
                                          S(ti_i, ti_a));

  // \f$L_{ai} = S_{ia}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_S_ia(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_i>(make_not_null(&Lai_from_S_ia),
                                          S(ti_i, ti_a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t i = 0; i < dim; i++) {
      CHECK(Lai_from_R_ai.get(a, i + 1) == R.get(a, i));
      CHECK(Lia_from_R_ai.get(i + 1, a) == R.get(a, i));
      CHECK(Lia_from_S_ia.get(i + 1, a) == S.get(i, a));
      CHECK(Lai_from_S_ia.get(a, i + 1) == S.get(i, a));
    }
  }
}

// \brief Test evaluation of rank 2 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType, typename Generator>
void test_rhs_and_lhs_rank2(
    const DataType& used_for_size,
    const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ai(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_i>(make_not_null(&Lai_from_R_ai),
                                          R(ti_a, ti_i));

  // \f$L_{ia} = R_{ai}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ai(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_a>(make_not_null(&Lia_from_R_ai),
                                          R(ti_a, ti_i));

  // \f$L_{ai} = R_{ia}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_ia(used_for_size);
  TensorExpressions::evaluate<ti_a, ti_i>(make_not_null(&Lai_from_R_ia),
                                          R(ti_i, ti_a));

  // \f$L_{ia} = R_{ia}\f$
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_ia(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_a>(make_not_null(&Lia_from_R_ia),
                                          R(ti_i, ti_a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t i = 0; i < dim; i++) {
      CHECK(Lai_from_R_ai.get(a, i + 1) == R.get(a, i + 1));
      CHECK(Lia_from_R_ai.get(i + 1, a) == R.get(a, i + 1));
      CHECK(Lai_from_R_ia.get(a, i + 1) == R.get(i + 1, a));
      CHECK(Lia_from_R_ia.get(i + 1, a) == R.get(i + 1, a));
    }
  }
}

// \brief Test evaluation of rank 4 tensors where generic spatial indices are
// used for RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename DataType, typename Generator>
void test_rhs_and_lhs_rank4(
    const DataType& used_for_size,
    const gsl::not_null<Generator*> generator) noexcept {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<3, 2, 1, 2>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpatialIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>>>>(
      generator, distribution, used_for_size);

  // \f$L_{ai} = R_{ai}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<dim, UpLo::Lo, Frame::Grid>>>
      Likaj_from_R_jaik(used_for_size);
  TensorExpressions::evaluate<ti_i, ti_k, ti_a, ti_j>(
      make_not_null(&Likaj_from_R_jaik), R(ti_j, ti_a, ti_i, ti_k));

  for (size_t i = 0; i < dim; i++) {
    for (size_t k = 0; k < dim; k++) {
      for (size_t a = 0; a < dim + 1; a++) {
        for (size_t j = 0; j < dim; j++) {
          CHECK(Likaj_from_R_jaik.get(i, k + 1, a, j) ==
                R.get(j + 1, a, i, k + 1));
        }
      }
    }
  }
}

template <typename DataType>
void test_evaluate_spatial_spacetime_index(
    const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);

  test_rhs(used_for_size, make_not_null(&generator));
  test_lhs(used_for_size, make_not_null(&generator));
  test_rhs_and_lhs_rank2(used_for_size, make_not_null(&generator));
  test_rhs_and_lhs_rank4(used_for_size, make_not_null(&generator));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.Expression.EvaluateSpatialSpacetimeIndex",
    "[DataStructures][Unit]") {
  test_evaluate_spatial_spacetime_index(
      std::numeric_limits<double>::signaling_NaN());
  test_evaluate_spatial_spacetime_index(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
