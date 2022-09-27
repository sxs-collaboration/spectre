// Distributed under the MIT License.
// See LICENSE.txt for details.

// \file
// Tests evaluation of `TensorExpression`s with complex-valued LHS `Tensor`s
// and RHS expressions that may contain real-valued terms, complex-valued terms,
// or both
//
// \details
// The tests in this file are designed to test that these fundamentally work:
// - Using `evaluate` with complex types on the RHS and/or LHS
// - Using `TensorExpression` mathematical operations with complex types
// - Evaluating expressions with both real-valued and complex-valued terms

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <complex>
#include <cstddef>
#include <random>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename T1, typename T2>
void check_values_equal(const T1& lhs_value, const T2& rhs_value) {
  CHECK_ITERABLE_APPROX(lhs_value, rhs_value);
}

template <>
void check_values_equal<std::complex<double>, double>(
    const std::complex<double>& lhs_value, const double& rhs_value) {
  CHECK(std::imag(lhs_value) == 0.0);
  CHECK_ITERABLE_APPROX(std::real(lhs_value), rhs_value);
}

template <>
void check_values_equal<ComplexDataVector, double>(
    const ComplexDataVector& lhs_value, const double& rhs_value) {
  for (size_t i = 0; i < lhs_value.size(); i++) {
    CHECK(std::imag(lhs_value[i]) == 0.0);
    CHECK_ITERABLE_APPROX(std::real(lhs_value[i]), rhs_value);
  }
}

template <>
void check_values_equal<ComplexDataVector, DataVector>(
    const ComplexDataVector& lhs_value, const DataVector& rhs_value) {
  for (size_t i = 0; i < lhs_value.size(); i++) {
    CHECK(std::imag(lhs_value[i]) == 0.0);
    CHECK_ITERABLE_APPROX(std::real(lhs_value[i]), rhs_value[i]);
  }
}

// \brief Test assignment of LHS `Tensor` to single RHS term
//
// \tparam LhsDataType the data type of LHS `Tensor`
// \tparam RhsDataType the data type of the RHS term
template <typename Generator, typename LhsDataType, typename RhsDataType>
void test_assignment_to_single_term(const gsl::not_null<Generator*> generator,
                                    const LhsDataType& used_for_size_lhs,
                                    const RhsDataType& used_for_size_rhs) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  // if the RHS is a number, also test the assignment of LHS to the number
  if constexpr (tenex::detail::is_supported_number_datatype_v<RhsDataType>) {
    const auto R1 = make_with_random_values<RhsDataType>(
        generator, distribution, used_for_size_rhs);
    Scalar<LhsDataType> L1{used_for_size_lhs};
    tenex::evaluate(make_not_null(&L1), R1);
    check_values_equal(get(L1), R1);
  }

  // assign Scalar<LhsDataType> to Scalar<RhsDataType>
  const auto R2 = make_with_random_values<Scalar<RhsDataType>>(
      generator, distribution, used_for_size_rhs);
  Scalar<LhsDataType> L2{used_for_size_lhs};
  tenex::evaluate(make_not_null(&L2), R2());
  check_values_equal(get(L2), get(R2));

  // assign Tensor<LhsDataType, ...> to Tensor<RhsDataType, ...>
  const auto R3 = make_with_random_values<tnsr::ij<RhsDataType, 3>>(
      generator, distribution, used_for_size_rhs);
  tnsr::ii<LhsDataType, 3> L3{used_for_size_lhs};
  tenex::evaluate<ti::j, ti::i>(make_not_null(&L3), R3(ti::i, ti::j));
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = i; j < 3; j++) {
      check_values_equal(L3.get(j, i), R3.get(i, j));
    }
  }
}

// \brief Test assignment of LHS `Tensor` to a RHS expression containing
// mathematical operations
//
// \tparam LhsDataType the data type of LHS `Tensor`
// \tparam RhsDataType the data type of the RHS term
template <typename Generator, typename LhsDataType, typename RhsDataType>
void test_evaluate_ops(const gsl::not_null<Generator*> generator,
                       const LhsDataType& used_for_size_lhs,
                       const RhsDataType& used_for_size_rhs) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto R =
      make_with_random_values<tnsr::Ab<RhsDataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size_rhs);
  const auto S =
      make_with_random_values<tnsr::aB<RhsDataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size_rhs);
  const auto T = make_with_random_values<Scalar<RhsDataType>>(
      generator, distribution, used_for_size_rhs);

  // test evaluation of unary ops
  Scalar<LhsDataType> L_contraction{used_for_size_lhs};
  tenex::evaluate(make_not_null(&L_contraction), R(ti::A, ti::a));
  RhsDataType expected_L_contraction =
      make_with_value<RhsDataType>(used_for_size_rhs, 0.0);
  for (size_t a = 0; a < 4; a++) {
    expected_L_contraction += R.get(a, a);
  }
  check_values_equal(get(L_contraction), expected_L_contraction);

  tnsr::Ab<LhsDataType, 3, Frame::Inertial> L_negation{used_for_size_lhs};
  tenex::evaluate<ti::B, ti::a>(make_not_null(&L_negation), -S(ti::a, ti::B));
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      check_values_equal(L_negation.get(b, a), -S.get(a, b));
    }
  }

  Scalar<LhsDataType> L_square_root{used_for_size_lhs};
  tenex::evaluate(make_not_null(&L_square_root), sqrt(T()));
  check_values_equal(get(L_square_root), sqrt(get(T)));

  // test evaluation of binary ops
  tnsr::Ab<LhsDataType, 3, Frame::Inertial> L_addition{used_for_size_lhs};
  tnsr::Ab<LhsDataType, 3, Frame::Inertial> L_subtraction{used_for_size_lhs};
  tenex::evaluate<ti::A, ti::b>(make_not_null(&L_addition),
                                R(ti::A, ti::b) + S(ti::b, ti::A));
  tenex::evaluate<ti::B, ti::a>(make_not_null(&L_subtraction),
                                S(ti::a, ti::B) - R(ti::B, ti::a));
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      check_values_equal(L_addition.get(a, b), R.get(a, b) + S.get(b, a));
      check_values_equal(L_subtraction.get(b, a), S.get(a, b) - R.get(b, a));
    }
  }

  tnsr::aB<LhsDataType, 3, Frame::Inertial> L_product{used_for_size_lhs};
  tenex::evaluate<ti::a, ti::B>(make_not_null(&L_product),
                                R(ti::C, ti::a) * S(ti::c, ti::B));
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      RhsDataType expected_sum =
          make_with_value<RhsDataType>(used_for_size_rhs, 0.0);
      for (size_t c = 0; c < 4; c++) {
        expected_sum += R.get(c, a) * S.get(c, b);
      }
      check_values_equal(L_product.get(a, b), expected_sum);
    }
  }

  tnsr::aB<LhsDataType, 3, Frame::Inertial> L_division{used_for_size_lhs};
  tenex::evaluate<ti::a, ti::B>(make_not_null(&L_division),
                                S(ti::a, ti::B) / T());
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      check_values_equal(L_division.get(a, b), S.get(a, b) / get(T));
    }
  }
}

// \brief Test evaluation of RHS binary operations between real-valued and
// complex-valued terms
//
// \details
// Tests when (1) the terms are both `Tensor`s and (2) when one term is a
// `Tensor` and the other is a number
//
// Note: Binary operations between a complex number and a
// `Tensor<DataVector, ...>` are only tested for multiplication. This is because
// for `std::complex<double> OP DataVector`, Blaze currently only supports
// multiplication.
//
// \tparam ComplexDataType the data type of the complex-valued operand
// \tparam RhsDataType the data type of the real-valued operand
template <typename Generator, typename ComplexDataType, typename RealDataType>
void test_bin_ops_with_real_and_complex(
    const gsl::not_null<Generator*> generator,
    const ComplexDataType& used_for_size_complex,
    const RealDataType& used_for_size_real,
    const double used_for_random_real_number,
    const std::complex<double> used_for_random_complex_number) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t Dim = 2;

  // Operands for test expressions

  const auto real_Ij =
      make_with_random_values<tnsr::Ij<RealDataType, Dim, Frame::Grid>>(
          generator, distribution, used_for_size_real);
  const auto complex_Ij =
      make_with_random_values<tnsr::Ij<ComplexDataType, Dim, Frame::Grid>>(
          generator, distribution, used_for_size_complex);
  const auto real_iJ =
      make_with_random_values<tnsr::iJ<RealDataType, Dim, Frame::Grid>>(
          generator, distribution, used_for_size_real);
  const auto complex_iJ =
      make_with_random_values<tnsr::iJ<ComplexDataType, Dim, Frame::Grid>>(
          generator, distribution, used_for_size_complex);
  const auto real_scalar = make_with_random_values<Scalar<RealDataType>>(
      generator, distribution, used_for_size_real);
  const auto complex_scalar = make_with_random_values<Scalar<ComplexDataType>>(
      generator, distribution, used_for_size_complex);
  const auto real_number = make_with_random_values<double>(
      generator, distribution, used_for_random_real_number);
  const auto complex_number = make_with_random_values<std::complex<double>>(
      generator, distribution, used_for_random_complex_number);

  // Tested expressions

  // addition
  const Scalar<ComplexDataType> complex_tensor_plus_real_number =
      tenex::evaluate(complex_scalar() + real_number);
  const Scalar<ComplexDataType> real_number_plus_complex_tensor =
      tenex::evaluate(real_number + complex_scalar());
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      complex_tensor_plus_real_tensor = tenex::evaluate<ti::i, ti::J>(
          complex_iJ(ti::i, ti::J) + real_iJ(ti::i, ti::J));
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      real_tensor_plus_complex_tensor = tenex::evaluate<ti::i, ti::J>(
          real_Ij(ti::J, ti::i) + complex_Ij(ti::J, ti::i));

  // subtraction
  const Scalar<ComplexDataType> complex_tensor_minus_real_number =
      tenex::evaluate(complex_scalar() - real_number);
  const Scalar<ComplexDataType> real_number_minus_complex_tensor =
      tenex::evaluate(real_number - complex_scalar());
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      complex_tensor_minus_real_tensor = tenex::evaluate<ti::i, ti::J>(
          complex_iJ(ti::i, ti::J) - real_iJ(ti::i, ti::J));
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      real_tensor_minus_complex_tensor = tenex::evaluate<ti::i, ti::J>(
          real_Ij(ti::J, ti::i) - complex_Ij(ti::J, ti::i));

  // multiplication
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      complex_tensor_times_real_number =
          tenex::evaluate<ti::i, ti::J>(complex_iJ(ti::i, ti::J) * real_number);
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      real_number_times_complex_tensor =
          tenex::evaluate<ti::i, ti::J>(real_number * complex_Ij(ti::J, ti::i));
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      real_tensor_times_complex_number =
          tenex::evaluate<ti::i, ti::J>(real_iJ(ti::i, ti::J) * complex_number);
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      complex_number_times_real_tensor =
          tenex::evaluate<ti::i, ti::J>(complex_number * real_iJ(ti::i, ti::J));
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      complex_tensor_times_real_tensor = tenex::evaluate<ti::i, ti::J>(
          complex_iJ(ti::i, ti::K) * real_iJ(ti::k, ti::J));
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      real_tensor_times_complex_tensor = tenex::evaluate<ti::i, ti::J>(
          real_Ij(ti::K, ti::i) * complex_Ij(ti::J, ti::k));

  // division
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      complex_tensor_over_real_number =
          tenex::evaluate<ti::i, ti::J>(complex_iJ(ti::i, ti::J) / real_number);
  const Scalar<ComplexDataType> real_number_over_complex_tensor =
      tenex::evaluate(real_number / complex_scalar());
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      complex_tensor_over_real_tensor = tenex::evaluate<ti::i, ti::J>(
          complex_iJ(ti::i, ti::J) / real_scalar());
  const tnsr::iJ<ComplexDataType, Dim, Frame::Grid>
      real_tensor_over_complex_tensor = tenex::evaluate<ti::i, ti::J>(
          real_Ij(ti::J, ti::i) / complex_scalar());

  // Check rank == 0 results

  // addition
  CHECK(get(complex_tensor_plus_real_number) ==
        get(complex_scalar) + real_number);
  CHECK(get(real_number_plus_complex_tensor) ==
        real_number + get(complex_scalar));

  // subtraction
  CHECK(get(complex_tensor_minus_real_number) ==
        get(complex_scalar) - real_number);
  CHECK(get(real_number_minus_complex_tensor) ==
        real_number - get(complex_scalar));

  // division
  CHECK(get(real_number_over_complex_tensor) ==
        real_number / get(complex_scalar));

  // Check rank > 0 results
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      ComplexDataType expected_sum_complex_tensor_times_real_tensor =
          complex_iJ.get(i, 0) * real_iJ.get(0, j);
      ComplexDataType expected_sum_real_tensor_times_complex_tensor =
          real_Ij.get(0, i) * complex_Ij.get(j, 0);
      for (size_t k = 1; k < Dim; k++) {
        expected_sum_complex_tensor_times_real_tensor +=
            complex_iJ.get(i, k) * real_iJ.get(k, j);
        expected_sum_real_tensor_times_complex_tensor +=
            real_Ij.get(k, i) * complex_Ij.get(j, k);
      }

      // addition
      CHECK(complex_tensor_plus_real_tensor.get(i, j) ==
            complex_iJ.get(i, j) + real_iJ.get(i, j));
      CHECK(real_tensor_plus_complex_tensor.get(i, j) ==
            real_Ij.get(j, i) + complex_Ij.get(j, i));

      // subtraction
      CHECK(complex_tensor_minus_real_tensor.get(i, j) ==
            complex_iJ.get(i, j) - real_iJ.get(i, j));
      CHECK(real_tensor_minus_complex_tensor.get(i, j) ==
            real_Ij.get(j, i) - complex_Ij.get(j, i));

      // multiplication
      CHECK(complex_tensor_times_real_number.get(i, j) ==
            complex_iJ.get(i, j) * real_number);
      CHECK(real_number_times_complex_tensor.get(i, j) ==
            real_number * complex_Ij.get(j, i));
      CHECK(real_tensor_times_complex_number.get(i, j) ==
            real_iJ.get(i, j) * complex_number);
      CHECK(complex_number_times_real_tensor.get(i, j) ==
            complex_number * real_iJ.get(i, j));
      CHECK_ITERABLE_APPROX(complex_tensor_times_real_tensor.get(i, j),
                            expected_sum_complex_tensor_times_real_tensor);
      CHECK_ITERABLE_APPROX(real_tensor_times_complex_tensor.get(i, j),
                            expected_sum_real_tensor_times_complex_tensor);

      // division
      CHECK_ITERABLE_APPROX(complex_tensor_over_real_number.get(i, j),
                            complex_iJ.get(i, j) / real_number);
      CHECK(complex_tensor_over_real_tensor.get(i, j) ==
            complex_iJ.get(i, j) / get(real_scalar));
      CHECK(real_tensor_over_complex_tensor.get(i, j) ==
            real_Ij.get(j, i) / get(complex_scalar));
    }
  }
}

// \brief Test evaluation of large RHS `TensorExpression`s
//
// \details
// Test cases include large expressions with:
// - only real-valued `Tensor`s
// - only complex-valued `Tensor`s
// - real-valued and complex-valued `Tensor`s
// - real-valued `Tensor`s and a real-valued number
// - complex-valued `Tensor`s and a complex-valued number (see note below)
// - real-valued `Tensor`s and a complex-valued number
// - complex-valued `Tensor`s and a real-valued number
//
// Note: Binary operations between a complex number and a
// `Tensor<DataVector, ...>` are only tested for multiplication. This is because
// for `std::complex<double> OP DataVector`, Blaze currently only supports
// multiplication.
//
// \tparam ComplexDataType the data type of the complex-valued operand
// \tparam RhsDataType the data type of the real-valued operand
template <typename Generator, typename ComplexDataType, typename RealDataType>
void test_evaluate_large_expressions(
    const gsl::not_null<Generator*> generator,
    const ComplexDataType& used_for_size_complex,
    const RealDataType& used_for_size_real,
    const double used_for_random_real_number,
    const std::complex<double> used_for_random_complex_number) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  // Operands for test expressions

  const auto real_scalar = make_with_random_values<Scalar<RealDataType>>(
      generator, distribution, used_for_size_real);
  const auto complex_scalar = make_with_random_values<Scalar<ComplexDataType>>(
      generator, distribution, used_for_size_complex);
  const auto real_number = make_with_random_values<double>(
      generator, distribution, used_for_random_real_number);
  const auto complex_number = make_with_random_values<std::complex<double>>(
      generator, distribution, used_for_random_complex_number);

  // Tested expressions

  const auto real_scalar_times_8 =
      real_scalar() + real_scalar() + real_scalar() + real_scalar() +
      real_scalar() + real_scalar() + real_scalar() + real_scalar();
  const auto real_scalar_times_64 = real_scalar_times_8 + real_scalar_times_8 +
                                    real_scalar_times_8 + real_scalar_times_8 +
                                    real_scalar_times_8 + real_scalar_times_8 +
                                    real_scalar_times_8 + real_scalar_times_8;

  const auto complex_scalar_times_8 = complex_scalar() + complex_scalar() +
                                      complex_scalar() + complex_scalar() +
                                      complex_scalar() + complex_scalar() +
                                      complex_scalar() + complex_scalar();
  const auto complex_scalar_times_64 =
      complex_scalar_times_8 + complex_scalar_times_8 + complex_scalar_times_8 +
      complex_scalar_times_8 + complex_scalar_times_8 + complex_scalar_times_8 +
      complex_scalar_times_8 + complex_scalar_times_8;

  // large expressions of `Tensor`s
  const Scalar<RealDataType> real_tensor_times_real_tensor_result =
      tenex::evaluate(real_scalar_times_64);
  const Scalar<ComplexDataType> complex_tensor_times_complex_tensor_result =
      tenex::evaluate(complex_scalar_times_64);
  const Scalar<ComplexDataType> complex_tensor_times_real_tensor_result =
      tenex::evaluate(complex_scalar_times_64 * real_scalar_times_64);
  const Scalar<ComplexDataType> real_tensor_times_complex_tensor_result =
      tenex::evaluate(real_scalar_times_64 * complex_scalar_times_64);

  // large expressions of `Tensor`s and a number
  const Scalar<RealDataType> real_tensor_times_real_number_result =
      tenex::evaluate(real_scalar_times_64 * real_number);
  const Scalar<RealDataType> real_number_times_real_tensor_result =
      tenex::evaluate(real_number * real_scalar_times_64);
  const Scalar<ComplexDataType> complex_tensor_times_real_number_result =
      tenex::evaluate(complex_scalar_times_64 * real_number);
  const Scalar<ComplexDataType> real_number_times_complex_tensor_result =
      tenex::evaluate(real_number * complex_scalar_times_64);
  const Scalar<ComplexDataType> real_tensor_times_complex_number_result =
      tenex::evaluate(real_scalar_times_64 * complex_number);
  const Scalar<ComplexDataType> complex_number_times_real_tensor_result =
      tenex::evaluate(complex_number * real_scalar_times_64);
  const Scalar<ComplexDataType> complex_tensor_times_complex_number_result =
      tenex::evaluate(complex_scalar_times_64 * complex_number);
  const Scalar<ComplexDataType> complex_number_times_complex_tensor_result =
      tenex::evaluate(complex_number * complex_scalar_times_64);

  // check expressions with only `Tensor`s
  CHECK_ITERABLE_APPROX(get(real_tensor_times_real_tensor_result),
                        64.0 * get(real_scalar));
  CHECK_ITERABLE_APPROX(get(complex_tensor_times_complex_tensor_result),
                        64.0 * get(complex_scalar));
  CHECK_ITERABLE_APPROX(get(complex_tensor_times_real_tensor_result),
                        64.0 * 64.0 * (get(complex_scalar) * get(real_scalar)));
  CHECK_ITERABLE_APPROX(get(real_tensor_times_complex_tensor_result),
                        64.0 * 64.0 * (get(complex_scalar) * get(real_scalar)));

  // check expressions with `Tensor`s and numbers
  CHECK_ITERABLE_APPROX(get(real_tensor_times_real_number_result),
                        64.0 * get(real_scalar) * real_number);
  CHECK_ITERABLE_APPROX(get(real_number_times_real_tensor_result),
                        64.0 * get(real_scalar) * real_number);
  CHECK_ITERABLE_APPROX(get(complex_tensor_times_real_number_result),
                        64.0 * get(complex_scalar) * real_number);
  CHECK_ITERABLE_APPROX(get(real_number_times_complex_tensor_result),
                        64.0 * get(complex_scalar) * real_number);
  CHECK_ITERABLE_APPROX(get(real_tensor_times_complex_number_result),
                        64.0 * get(real_scalar) * complex_number);
  CHECK_ITERABLE_APPROX(get(complex_number_times_real_tensor_result),
                        64.0 * get(real_scalar) * complex_number);
  CHECK_ITERABLE_APPROX(get(complex_tensor_times_complex_number_result),
                        64.0 * get(complex_scalar) * complex_number);
  CHECK_ITERABLE_APPROX(get(complex_number_times_complex_tensor_result),
                        64.0 * get(complex_scalar) * complex_number);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateComplex",
                  "[Unit][DataStructures]") {
  MAKE_GENERATOR(generator);

  const size_t vector_size = 2;

  const double used_for_size_real_double =
      std::numeric_limits<double>::signaling_NaN();
  const std::complex<double> used_for_size_complex_double =
      std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
                           std::numeric_limits<double>::signaling_NaN());
  const DataVector used_for_size_real_datavector =
      DataVector(vector_size, std::numeric_limits<double>::signaling_NaN());
  const ComplexDataVector used_for_size_complex_datavector = ComplexDataVector(
      vector_size, std::numeric_limits<double>::signaling_NaN());

  // Test assignment of complex-valued LHS `Tensor` to single RHS term
  test_assignment_to_single_term(make_not_null(&generator),
                                 used_for_size_complex_double,
                                 used_for_size_real_double);
  test_assignment_to_single_term(make_not_null(&generator),
                                 used_for_size_complex_double,
                                 used_for_size_complex_double);
  test_assignment_to_single_term(make_not_null(&generator),
                                 used_for_size_complex_datavector,
                                 used_for_size_real_double);
  test_assignment_to_single_term(make_not_null(&generator),
                                 used_for_size_complex_datavector,
                                 used_for_size_real_datavector);
  test_assignment_to_single_term(make_not_null(&generator),
                                 used_for_size_complex_datavector,
                                 used_for_size_complex_datavector);

  // Test assignment of a complex-valued LHS `Tensor` to a RHS expression
  // containing mathematical operations
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_double,
                    used_for_size_real_double);
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_double,
                    used_for_size_complex_double);
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_datavector,
                    used_for_size_real_double);
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_datavector,
                    used_for_size_real_datavector);
  test_evaluate_ops(make_not_null(&generator), used_for_size_complex_datavector,
                    used_for_size_complex_datavector);

  // Test evaluation of RHS binary operations between real-valued and
  // complex-valued terms
  test_bin_ops_with_real_and_complex(
      make_not_null(&generator), used_for_size_complex_double,
      used_for_size_real_double, used_for_size_real_double,
      used_for_size_complex_double);
  test_bin_ops_with_real_and_complex(
      make_not_null(&generator), used_for_size_complex_datavector,
      used_for_size_real_datavector, used_for_size_real_double,
      used_for_size_complex_double);

  // Test evaluation of large RHS `TensorExpression`s
  test_evaluate_large_expressions(
      make_not_null(&generator), used_for_size_complex_double,
      used_for_size_real_double, used_for_size_real_double,
      used_for_size_complex_double);
  test_evaluate_large_expressions(
      make_not_null(&generator), used_for_size_complex_datavector,
      used_for_size_real_datavector, used_for_size_real_double,
      used_for_size_complex_double);
}
