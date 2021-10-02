// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// \brief Test the division of a tensor expression over a `double` is correctly
// evaluated
//
// \details
// The cases tested are:
// - \f$L_{ij} = S_{ij} / R\f$
// - \f$L_{ij} = (R * S_{ij} / T) / G\f$
//
// where \f$R\f$, \f$T\f$, and \f$G\f$ are `double`s and \f$S\f$ and \f$L\f$ are
// Tensors with data type `double` or DataVector.
//
// \tparam DataType the type of data being stored in the numerator
template <typename Generator, typename DataType>
void test_divide_double_denominator(const gsl::not_null<Generator*> generator,
                                    const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(-2.0, 2.0);
  constexpr size_t dim = 3;

  const auto S = make_with_random_values<tnsr::ii<DataType, dim>>(
      generator, distribution, used_for_size);

  // \f$L_{ij} = S_{ij} / R\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const tnsr::ii<DataType, dim> Lij_from_Sij_over_R =
      TensorExpressions::evaluate<ti_i, ti_j>(S(ti_i, ti_j) / 4.3);
  // \f$L_{ij} = (R * S_{ij} / T / G)\f$
  const tnsr::ii<DataType, dim> Lij_from_R_Sij_over_T =
      TensorExpressions::evaluate<ti_i, ti_j>((-5.2 * S(ti_i, ti_j) / 1.6) /
                                              2.1);

  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      CHECK_ITERABLE_APPROX(Lij_from_Sij_over_R.get(i, j), S.get(i, j) / 4.3);
      CHECK_ITERABLE_APPROX(Lij_from_R_Sij_over_T.get(i, j),
                            -5.2 * S.get(i, j) / 1.6 / 2.1);
    }
  }
}

// \brief Test the division of a `double` over a tensor expression is correctly
// evaluated
//
// \details
// The cases tested are:
// - \f$L = R / S\f$
// - \f$L = R / \sqrt{T^j{}_j}\f$
//
// where \f$R\f$ is a `double` and \f$S\f$, \f$T\f$, and \f$L\f$ are tensors
// with data type `double` or DataVector.
//
// \tparam DataType the type of data being stored in the numerator
template <typename Generator, typename DataType>
void test_divide_double_numerator(const gsl::not_null<Generator*> generator,
                                  const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 2.0);
  constexpr size_t dim = 3;

  const auto S = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto T = make_with_random_values<tnsr::Ij<DataType, dim>>(
      generator, distribution, used_for_size);

  // \f$L = R / S\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Scalar<DataType> result1 = TensorExpressions::evaluate(2.1 / S());
  CHECK(get(result1) == 2.1 / get(S));

  // \f$L = R / \sqrt{T^j{}_j}\f$
  const Scalar<DataType> result2 =
      TensorExpressions::evaluate(-5.7 / sqrt(T(ti_J, ti_j)));

  DataType trace_T = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t j = 0; j < dim; j++) {
    trace_T += T.get(j, j);
  }
  CHECK_ITERABLE_APPROX(get(result2), -5.7 / sqrt(trace_T));
}

// \brief Test the division of a tensor expression over a rank 0 tensor
// expression is correctly evaluated
//
// \details
// The cases tested are:
// - \f$L_{i}{}^{j} = S^{i}{}_{j} / R\f$
// - \f$L^{k}{}_{i} = (R T_{i}{}^{k}) / (T_{j}{}^{l} S^{j}{}_{l})\f$
// - \f$L_{i}{}^{k} = T_{i}{}^{k} / R^2 / R\f$
//
// where \f$R\f$, \f$S\f$, \f$T\f$, and \f$L\f$ are Tensors with data type
// `double` or DataVector.
//
// \tparam DataType the type of data being stored in the operands and result
template <typename Generator, typename DataType>
void test_divide_rank0_denominator(const gsl::not_null<Generator*> generator,
                                   const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 2.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto S = make_with_random_values<tnsr::Ij<DataType, dim>>(
      generator, distribution, used_for_size);
  const auto T = make_with_random_values<tnsr::iJ<DataType, dim>>(
      generator, distribution, used_for_size);

  // \f$L_{i}{}^{j} = S^{i}{}_{j} / R\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const tnsr::iJ<DataType, dim> result1 =
      TensorExpressions::evaluate<ti_j, ti_I>(S(ti_I, ti_j) / R());

  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      CHECK(result1.get(j, i) == S.get(i, j) / get(R));
    }
  }

  // \f$L^{k}{}_{i} = (R T_{i}{}^{k}) / (T_{j}{}^{l} S^{j}{}_{l})\f$
  const tnsr::Ij<DataType, dim> result2 =
      TensorExpressions::evaluate<ti_K, ti_i>(
          (R() * T(ti_i, ti_K)) / ((T(ti_j, ti_L) * S(ti_J, ti_l))));

  DataType result2_expected_denominator =
      make_with_value<DataType>(used_for_size, 0.0);
  for (size_t j = 0; j < dim; j++) {
    for (size_t l = 0; l < dim; l++) {
      result2_expected_denominator += T.get(j, l) * S.get(j, l);
    }
  }

  for (size_t i = 0; i < dim; i++) {
    for (size_t k = 0; k < dim; k++) {
      CHECK_ITERABLE_APPROX(
          result2.get(k, i),
          get(R) * T.get(i, k) / result2_expected_denominator);
    }
  }

  // \f$L_{i}{}^{k} = T_{i}{}^{k} / R^2 / R\f$
  const tnsr::iJ<DataType, dim> result3 =
      TensorExpressions::evaluate<ti_i, ti_K>(T(ti_i, ti_K) / square(R()) /
                                              R());

  DataType result3_expected_denominator = get(R) * get(R) * get(R);
  for (size_t i = 0; i < dim; i++) {
    for (size_t k = 0; k < dim; k++) {
      CHECK_ITERABLE_APPROX(result3.get(i, k),
                            T.get(i, k) / result3_expected_denominator);
    }
  }
}

// \brief Test the division of a tensor expression over a rank 0 tensor
// expression is correctly evaluated when generic indices are used for some
// spacetime indices
//
// \details
// The cases tested are:
// - \f$L_{ai} = (T_{a}{}^{i} - S^{i}{}_{a}) / R\f$
// - \f$L = (R / (T_{j}{}^{l} S^{j}{}_{l}) / 2\f$
//
// where \f$R\f$, \f$S\f$, \f$T\f$, and \f$L\f$ are Tensors with data type
// `double` or DataVector.
//
// \tparam DataType the type of data being stored in the operands and result
template <typename Generator, typename DataType>
void test_divide_spatial_spacetime_index(
    const gsl::not_null<Generator*> generator, const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 2.0);
  constexpr size_t dim = 3;
  using aI = Tensor<DataType, Symmetry<2, 1>,
                    index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                               SpatialIndex<dim, UpLo::Up, Frame::Inertial>>>;

  const auto R = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto S = make_with_random_values<tnsr::Ab<DataType, dim>>(
      generator, distribution, used_for_size);
  const auto T =
      make_with_random_values<aI>(generator, distribution, used_for_size);

  // \f$L_{ai} = (T_{a}{}^{i} - S^{i}{}_{a}) / R\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const aI result1 = TensorExpressions::evaluate<ti_a, ti_I>(
      (T(ti_a, ti_I) - S(ti_I, ti_a)) / R());

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t i = 0; i < dim; i++) {
      CHECK_ITERABLE_APPROX(result1.get(a, i),
                            (T.get(a, i) - S.get(i + 1, a)) / get(R));
    }
  }

  // \f$L = (R / (T_{j}{}^{l} S^{j}{}_{l}) / 2\f$
  const Scalar<DataType> result2 =
      TensorExpressions::evaluate(R() / (T(ti_j, ti_L) * S(ti_J, ti_l)) / 2.0);

  DataType result2_expected_denominator =
      make_with_value<DataType>(used_for_size, 0.0);
  for (size_t j = 0; j < dim; j++) {
    for (size_t l = 0; l < dim; l++) {
      result2_expected_denominator += T.get(j + 1, l) * S.get(j + 1, l + 1);
    }
  }

  CHECK_ITERABLE_APPROX(get(result2),
                        0.5 * get(R) / result2_expected_denominator);
}

// \brief Test the division of a tensor expression over a rank 0 tensor
// expression is correctly evaluated when time indices are used for some
// spacetime indices
//
// \details
// The cases tested are:
// - \f$L^{a} = S^{a}{}_{t} / \sqrt{R}\f$
// - \f$L = R / (T_{tt} / S^{t}{}_{t} \f$
//
// where \f$R\f$, \f$S\f$, \f$T\f$, and \f$L\f$ are Tensors with data type
// `double` or DataVector.
//
// \tparam DataType the type of data being stored in the operands and result
template <typename Generator, typename DataType>
void test_divide_time_index(const gsl::not_null<Generator*> generator,
                            const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 2.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto S = make_with_random_values<tnsr::Ab<DataType, dim>>(
      generator, distribution, used_for_size);
  const auto T = make_with_random_values<tnsr::ab<DataType, dim>>(
      generator, distribution, used_for_size);

  // \f$L^{a} = S^{a}{}_{t} / \sqrt{R}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const tnsr::A<DataType, dim> result1 =
      TensorExpressions::evaluate<ti_A>((S(ti_A, ti_t)) / sqrt(R()));

  for (size_t a = 0; a < dim + 1; a++) {
    CHECK_ITERABLE_APPROX(result1.get(a), (S.get(a, 0)) / sqrt(get(R)));
  }

  // \f$L = R / (T_{tt} / S^{t}{}_{t} \f$
  const Scalar<DataType> result2 =
      TensorExpressions::evaluate(R() / T(ti_t, ti_t) / S(ti_T, ti_t));

  CHECK_ITERABLE_APPROX(get(result2), get(R) / T.get(0, 0) / S.get(0, 0));
}

template <typename Generator, typename DataType>
void test_divide(const gsl::not_null<Generator*> generator,
                 const DataType& used_for_size) {
  test_divide_double_denominator(generator, used_for_size);
  test_divide_double_numerator(generator, used_for_size);
  test_divide_rank0_denominator(generator, used_for_size);
  test_divide_spatial_spacetime_index(generator, used_for_size);
  test_divide_time_index(generator, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Divide",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test_divide(make_not_null(&generator),
              std::numeric_limits<double>::signaling_NaN());
  test_divide(make_not_null(&generator),
              DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
