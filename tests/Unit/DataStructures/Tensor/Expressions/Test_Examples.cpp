// Distributed under the MIT License.
// See LICENSE.txt for details.

// \file
// Defines and tests examples for how to use `TensorExpression`s

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// test `tenex::evaluate()`
template <typename Generator, typename DataType>
void test_evaluate(const gsl::not_null<Generator*> generator,
                   const std::uniform_real_distribution<>& distribution,
                   const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  const auto R = make_with_random_values<tnsr::ab<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto g = make_with_random_values<tnsr::AA<DataType, Dim>>(
      generator, distribution, used_for_size);

  tnsr::Ab<DataType, Dim> expected_result{};
  for (size_t c = 0; c < Dim + 1; c++) {
    for (size_t b = 0; b < Dim + 1; b++) {
      expected_result.get(c, b) = R.get(0, b) * g.get(0, c);
      for (size_t a = 1; a < Dim + 1; a++) {
        expected_result.get(c, b) += R.get(a, b) * g.get(a, c);
      }
    }
  }

  {
    // [te_example_evaluate_lhs_return]
    auto R_up =
        tenex::evaluate<ti::C, ti::b>(R(ti::a, ti::b) * g(ti::A, ti::C));
    // [te_example_evaluate_lhs_return]
    CHECK_ITERABLE_APPROX(R_up, expected_result);
  }
  {
    // [te_example_evaluate_lhs_arg]
    tnsr::Ab<DataType, Dim> R_up{};
    tenex::evaluate<ti::C, ti::b>(make_not_null(&R_up),
                                  R(ti::a, ti::b) * g(ti::A, ti::C));
    // [te_example_evaluate_lhs_arg]
    CHECK_ITERABLE_APPROX(R_up, expected_result);
  }
}

// tests arithmetric operations
template <typename Generator, typename DataType>
void test_basic_operations(const gsl::not_null<Generator*> generator,
                           const std::uniform_real_distribution<>& distribution,
                           const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  const auto R = make_with_random_values<tnsr::ab<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto S = make_with_random_values<tnsr::ab<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto T = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);
  const auto U = make_with_random_values<tnsr::Ab<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto V = make_with_random_values<tnsr::aBC<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto G = make_with_random_values<tnsr::a<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto H = make_with_random_values<tnsr::A<DataType, Dim>>(
      generator, distribution, used_for_size);

  // some examples below are forced to be more than one line against the wishes
  // of clang-tidy because the doxygen \snippet function can't render
  // single-lined code snippets

  // addition
  {
    // [te_example_addition]
    auto L =
        tenex::evaluate<ti::a, ti::b>(R(ti::a, ti::b) + S(ti::b, ti::a));
    // [te_example_addition]

    tnsr::ab<DataType, Dim> expected_result{};
    for (size_t a = 0; a < Dim + 1; a++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        expected_result.get(a, b) = R.get(a, b) + S.get(b, a);
      }
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  // subtraction
  {
    // [te_example_subtraction]
    auto L =
        tenex::evaluate(1.0 - T());
    // [te_example_subtraction]

    const Scalar<DataType> expected_result{1.0 - get(T)};

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    // [te_example_contraction_to_scalar]
    auto L =
        tenex::evaluate(U(ti::A, ti::a));
    // [te_example_contraction_to_scalar]

    Scalar<DataType> expected_result{get<0, 0>(U)};
    for (size_t a = 1; a < Dim + 1; a++) {
      get(expected_result) += U.get(a, a);
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    // [te_example_contraction_to_tensor]
    auto L =
        tenex::evaluate<ti::B>(V(ti::a, ti::B, ti::A));
    // [te_example_contraction_to_tensor]

    tnsr::A<DataType, Dim> expected_result{};
    for (size_t b = 0; b < Dim + 1; b++) {
      expected_result.get(b) = V.get(0, b, 0);
      for (size_t a = 1; a < Dim + 1; a++) {
        expected_result.get(b) += V.get(a, b, a);
      }
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    // [te_example_inner_product]
    auto L =
        tenex::evaluate(G(ti::a) * H(ti::A));
    // [te_example_inner_product]

    Scalar<DataType> expected_result{get<0>(G) * get<0>(H)};
    for (size_t a = 1; a < Dim + 1; a++) {
      get(expected_result) += G.get(a) * H.get(a);
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    // [te_example_inner_and_outer_product]
    auto L = tenex::evaluate<ti::c, ti::b>(T() * G(ti::a) * G(ti::c) *
                                           U(ti::A, ti::b));
    // [te_example_inner_and_outer_product]

    tnsr::ab<DataType, Dim> expected_result{};
    for (size_t c = 0; c < Dim + 1; c++) {
      for (size_t b = 0; b < Dim + 1; b++) {
        expected_result.get(c, b) = get<0>(G) * G.get(c) * U.get(0, b);
        for (size_t a = 1; a < Dim + 1; a++) {
          expected_result.get(c, b) += G.get(a) * G.get(c) * U.get(a, b);
        }
        expected_result.get(c, b) *= get(T);
      }
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    // [te_example_division_by_number]
    auto L =
        tenex::evaluate<ti::a>(G(ti::a) / 2.0);
    // [te_example_division_by_number]

    tnsr::a<DataType, Dim> expected_result{};
    for (size_t a = 0; a < Dim + 1; a++) {
      expected_result.get(a) = G.get(a) / 2.0;
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    // [te_example_division_by_tensor]
    auto L =
        tenex::evaluate<ti::b, ti::a>(R(ti::a, ti::b) / T());
    // [te_example_division_by_tensor]

    tnsr::ab<DataType, Dim> expected_result{};
    for (size_t b = 0; b < Dim + 1; b++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        expected_result.get(b, a) = R.get(a, b) / get(T);
      }
    }

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    // [te_example_division_by_tensor_expression]
    auto L =
        tenex::evaluate(5.0 / (U(ti::A, ti::a) + 1.0));
    // [te_example_division_by_tensor_expression]

    auto expected_result =
        make_with_value<Scalar<DataType>>(used_for_size, 1.0);
    for (size_t a = 0; a < Dim + 1; a++) {
      get(expected_result) += U.get(a, a);
    }
    get(expected_result) = 5.0 / get(expected_result);

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  // square root
  {
    // [te_example_square_root_tensor]
    auto L =
        tenex::evaluate(sqrt(T()));
    // [te_example_square_root_tensor]

    const Scalar<DataType> expected_result{sqrt(get(T))};

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
  {
    // [te_example_square_root_inner_product]
    auto L =
        tenex::evaluate(sqrt(G(ti::a) * H(ti::A)));
    // [te_example_square_root_inner_product]

    Scalar<DataType> expected_result{get<0>(G) * get<0>(H)};
    for (size_t a = 1; a < Dim + 1; a++) {
      get(expected_result) += G.get(a) * H.get(a);
    }
    get(expected_result) = sqrt(get(expected_result));

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
}

// tests LHS symmetry deduction and override by user
template <typename Generator, typename DataType>
void test_specify_lhs_symmetry(
    const gsl::not_null<Generator*> generator,
    const std::uniform_real_distribution<>& distribution,
    const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  const auto R = make_with_random_values<tnsr::a<DataType, Dim>>(
      generator, distribution, used_for_size);

  tnsr::aa<DataType, Dim> expected_result{};
  for (size_t a = 0; a < Dim + 1; a++) {
    for (size_t b = a; b < Dim + 1; b++) {
      expected_result.get(a, b) = R.get(a) * R.get(b);
    }
  }

  {
    // [te_example_deduced_lhs_symmetry_fail]
    auto L = tenex::evaluate<ti::a, ti::b>(R(ti::a) * R(ti::b));
    static_assert(std::is_same_v<decltype(L), tnsr::ab<DataType, Dim>>);
    // [te_example_deduced_lhs_symmetry_fail]

    for (size_t a = 0; a < 4; a++) {
      for (size_t b = 0; b < 4; b++) {
        CHECK(L.get(a, b) == expected_result.get(a, b));
        CHECK(L.get(b, a) == expected_result.get(a, b));
      }
    }
  }
  {
    // [te_example_deduced_lhs_symmetry_force]
    tnsr::aa<DataType, 3> L{};
    tenex::evaluate<ti::a, ti::b>(make_not_null(&L), R(ti::a) * R(ti::b));
    // [te_example_deduced_lhs_symmetry_force]

    CHECK_ITERABLE_APPROX(L, expected_result);
  }
}

// tests assignment of a RHS number to a LHS tensor
template <typename DataType>
void test_assign_number() {
  static_assert(std::is_same_v<DataType, double> or
                std::is_same_v<DataType, DataVector>);

  constexpr size_t Dim = 3;

  if constexpr (std::is_same_v<DataType, double>) {
    // [te_example_assign_number_to_tensor_of_numbers]
    tnsr::ab<double, Dim> L{};
    tenex::evaluate<ti::a, ti::b>(make_not_null(&L), -1.0);
    // [te_example_assign_number_to_tensor_of_numbers]

    const auto expected_result = make_with_value<tnsr::ab<double, Dim>>(
        std::numeric_limits<double>::signaling_NaN(), -1.0);
    CHECK(L == expected_result);
  } else if constexpr (std::is_same_v<DataType, DataVector>) {
    // [te_example_assign_number_to_tensor_of_vectors]
    // construct LHS tensor with size 5 DataVector
    tnsr::ab<DataVector, Dim> L{DataVector(5, 0.0)};
    tenex::evaluate<ti::a, ti::b>(make_not_null(&L), -1.0);
    // [te_example_assign_number_to_tensor_of_vectors]

    const size_t num_points = L[0].size();
    const auto expected_result = make_with_value<tnsr::ab<DataVector, Dim>>(
        DataVector(num_points, std::numeric_limits<double>::signaling_NaN()),
        -1.0);
    CHECK(L == expected_result);
  }
}

// tests usage of spatial and time indices for spacetime indices on the RHS
template <typename Generator, typename DataType>
void test_rhs_spatial_and_time_indices(
    const gsl::not_null<Generator*> generator, const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  // use gr random helper functions and calculate spacetime metric explicitly to
  // avoid taking square root of a negative in the tested expression
  const Scalar<DataType> random_lapse =
      TestHelpers::gr::random_lapse(generator, used_for_size);
  const tnsr::I<DataType, Dim> random_shift =
      TestHelpers::gr::random_shift<Dim>(generator, used_for_size);
  const tnsr::ii<DataType, Dim> random_spatial_metric =
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size);

  auto spacetime_metric = make_with_value<tnsr::aa<DataType, Dim>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  gr::spacetime_metric(make_not_null(&spacetime_metric), random_lapse,
                       random_shift, random_spatial_metric);
  const auto& shift = random_shift;
  const auto& expected_result = random_lapse;

  // [te_example_rhs_spatial_and_time_indices]
  auto lapse =
      tenex::evaluate(sqrt(shift(ti::I) * spacetime_metric(ti::i, ti::t) -
                           spacetime_metric(ti::t, ti::t)));
  // [te_example_rhs_spatial_and_time_indices]

  CHECK_ITERABLE_APPROX(lapse, expected_result);
}

// tests usage of spatial and time indices for spacetime indices on the LHS
template <typename Generator, typename DataType>
void test_lhs_spatial_and_time_indices(
    const gsl::not_null<Generator*> generator,
    const std::uniform_real_distribution<>& distribution,
    const DataType& used_for_size) {
  constexpr size_t Dim = 3;

  const auto spatial_metric = make_with_random_values<tnsr::ii<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<DataType, Dim>>(
      generator, distribution, used_for_size);
  const auto lapse = make_with_random_values<Scalar<DataType>>(
      generator, distribution, used_for_size);

  // [te_example_lhs_spatial_and_time_indices]
  tnsr::aa<DataType, Dim> spacetime_metric{};
  tenex::evaluate<ti::t, ti::t>(
      make_not_null(&spacetime_metric),
      -square(lapse()) +
          shift(ti::M) * shift(ti::N) * spatial_metric(ti::m, ti::n));
  tenex::evaluate<ti::t, ti::i>(make_not_null(&spacetime_metric),
                                spatial_metric(ti::m, ti::i) * shift(ti::M));
  tenex::evaluate<ti::i, ti::j>(make_not_null(&spacetime_metric),
                                spatial_metric(ti::i, ti::j));
  // [te_example_lhs_spatial_and_time_indices]

  // note: there are more efficient ways to implement this equation, but
  // choosing the simplest to read and write
  tnsr::aa<DataType, 3> expected_result{};
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = i; j < Dim; j++) {
      expected_result.get(i + 1, j + 1) = spatial_metric.get(i, j);
    }
  }

  for (size_t i = 0; i < Dim; i++) {
    expected_result.get(0, i + 1) = spatial_metric.get(0, i) * shift.get(0);
    for (size_t m = 1; m < Dim; m++) {
      expected_result.get(0, i + 1) += spatial_metric.get(m, i) * shift.get(m);
    }
  }

  get<0, 0>(expected_result) = -square(get(lapse));
  for (size_t m = 0; m < Dim; m++) {
    for (size_t n = 0; n < Dim; n++) {
      get<0, 0>(expected_result) +=
          spatial_metric.get(m, n) * shift.get(m) * shift.get(n);
    }
  }

  CHECK_ITERABLE_APPROX(spacetime_metric, expected_result);
}

// tests using real-valued and complex-valued terms in the RHS expression
template <typename Generator, typename RealDataType>
void test_complex(const gsl::not_null<Generator*> generator,
                  const std::uniform_real_distribution<>& distribution,
                  const RealDataType& used_for_size) {
  static_assert(std::is_same_v<RealDataType, double> or
                std::is_same_v<RealDataType, DataVector>);

  constexpr size_t Dim = 3;

  if constexpr (std::is_same_v<RealDataType, double>) {
    // [te_example_complex_double]
    const auto x = make_with_random_values<tnsr::I<double, Dim>>(
        generator, distribution, used_for_size);
    const auto y = make_with_random_values<tnsr::I<double, Dim>>(
        generator, distribution, used_for_size);
    const std::complex<double> i{0.0, 1.0};

    const tnsr::I<std::complex<double>, Dim> z =
        tenex::evaluate<ti::I>(x(ti::I) + i * y(ti::I));
    // [te_example_complex_double]

    tnsr::I<std::complex<double>, Dim> expected_result{};

    get<0>(expected_result) = get<0>(x) + i * get<0>(y);
    for (size_t j = 1; j < Dim; j++) {
      expected_result.get(j) = x.get(j) + i * y.get(j);
    }

    CHECK_ITERABLE_APPROX(z, expected_result);
  } else if constexpr (std::is_same_v<RealDataType, DataVector>) {
    // [te_example_complex_vector]
    const auto x = make_with_random_values<tnsr::I<DataVector, Dim>>(
        generator, distribution, used_for_size);
    const auto y = make_with_random_values<tnsr::I<DataVector, Dim>>(
        generator, distribution, used_for_size);
    const std::complex<double> i{0.0, 1.0};

    const tnsr::I<ComplexDataVector, Dim> z =
        tenex::evaluate<ti::I>(x(ti::I) + i * y(ti::I));
    // [te_example_complex_vector]

    tnsr::I<ComplexDataVector, Dim> expected_result{};

    get<0>(expected_result) = get<0>(x) + i * get<0>(y);
    for (size_t j = 1; j < Dim; j++) {
      expected_result.get(j) = x.get(j) + i * y.get(j);
    }

    CHECK_ITERABLE_APPROX(z, expected_result);
  }
}

// runs all example tests (see individual functions)
template <typename Generator, typename DataType>
void test_examples(const gsl::not_null<Generator*> generator,
                   const std::uniform_real_distribution<>& distribution,
                   const DataType& used_for_size) {
  test_evaluate(generator, distribution, used_for_size);
  test_basic_operations(generator, distribution, used_for_size);
  test_specify_lhs_symmetry(generator, distribution, used_for_size);
  test_assign_number<DataType>();
  test_rhs_spatial_and_time_indices(generator, used_for_size);
  test_lhs_spatial_and_time_indices(generator, distribution, used_for_size);
  test_complex(generator, distribution, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Examples",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);
  const std::uniform_real_distribution<> distribution(0.1, 1.0);

  test_examples(make_not_null(&generator), distribution,
                std::numeric_limits<double>::signaling_NaN());
  test_examples(make_not_null(&generator), distribution,
                DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
