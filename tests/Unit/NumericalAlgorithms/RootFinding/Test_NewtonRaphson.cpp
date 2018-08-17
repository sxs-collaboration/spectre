// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/Exceptions.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
std::pair<double, double> func_and_deriv_free(double x) noexcept {
  return std::make_pair(2. - square(x), -2. * x);
}
struct FuncAndDeriv {
  std::pair<double, double> operator()(double x) noexcept {
    return std::make_pair(2. - square(x), -2. * x);
  }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.NewtonRaphson",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  /// [double_newton_raphson_root_find]
  const size_t digits = 8;
  const double correct = sqrt(2.);
  const double guess = 1.5;
  const double lower = 1.;
  const double upper = 2.;
  const auto func_and_deriv_lambda = [](double x) noexcept {
    return std::make_pair(2. - square(x), -2. * x);
  };
  const auto root_from_lambda = RootFinder::newton_raphson(
      func_and_deriv_lambda, guess, lower, upper, digits);
  /// [double_newton_raphson_root_find]
  const auto root_from_free = RootFinder::newton_raphson(
      func_and_deriv_free, guess, lower, upper, digits);
  const FuncAndDeriv func_and_deriv_functor{};
  const auto root_from_functor = RootFinder::newton_raphson(
      func_and_deriv_functor, guess, lower, upper, digits);
  CHECK(std::abs(root_from_lambda - correct) < 1.0 / std::pow(10, digits));
  CHECK(root_from_free == root_from_lambda);
  CHECK(root_from_free == root_from_functor);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.NewtonRaphson.Bounds",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  const size_t digits = 8;
  const double guess = 1.5;
  double upper = 2.;
  double lower = sqrt(2.);
  const auto func_and_deriv_lambda = [](double x) noexcept {
    return std::make_pair(2. - square(x), -2. * x);
  };

  auto root = RootFinder::newton_raphson(func_and_deriv_lambda, guess, lower,
                                         upper, digits);
  const double correct = sqrt(2.);

  CHECK(std::abs(root - correct) < 1.0 / std::pow(10, digits));

  lower = 0.;
  upper = sqrt(2.);

  root = RootFinder::newton_raphson(func_and_deriv_lambda, guess, lower, upper,
                                    digits);
  CHECK(std::abs(root - correct) < 1.0 / std::pow(10, digits));
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.NewtonRaphson.DataVector",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  /// [datavector_newton_raphson_root_find]
  const size_t digits = 8;
  const DataVector guess{1.6, 1.9, -1.6, -1.9};
  const DataVector lower{sqrt(2.), sqrt(2.), -2., -3.};
  const DataVector upper{2., 3., -sqrt(2.), -sqrt(2.)};
  const DataVector constant{2., 4., 2., 4.};

  const auto func_and_deriv_lambda = [&constant](const double x,
                                                 const size_t i) noexcept {
    return std::make_pair(constant[i] - square(x), -2. * x);
  };

  const auto root = RootFinder::newton_raphson(func_and_deriv_lambda, guess,
                                               lower, upper, digits);
  /// [datavector_newton_raphson_root_find]

  const DataVector correct{sqrt(2.), 2., -sqrt(2.), -2.};

  for (size_t i = 0; i < guess.size(); i++) {
    CHECK(std::abs(root[i] - correct[i]) < 1.0 / std::pow(10, digits));
  }
}

// [[OutputRegex, The desired accuracy of 100 base-10 digits must be smaller]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.NewtonRaphson.Digits.Double",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const size_t digits = 100;
  const double guess = 1.5;
  double lower = 1.;
  double upper = 2.;
  const auto func_and_deriv_lambda = [](double x) {
    return std::make_pair(2. - square(x), -2. * x);
  };

  RootFinder::newton_raphson(func_and_deriv_lambda, guess, lower, upper,
                             digits);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The desired accuracy of 100 base-10 digits must be smaller]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.NewtonRaphson.Digits.DataVector",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const size_t digits = 100;
  const DataVector guess{1.6, 1.9, -1.6, -1.9};
  const DataVector lower{sqrt(2.), sqrt(2.), -2., -3.};
  const DataVector upper{2., 3., -sqrt(2.), -sqrt(2.)};
  const DataVector constant{2., 4., 2., 4.};

  const auto func_and_deriv_lambda = [&constant](const double x,
                                                 const size_t i) noexcept {
    return std::make_pair(constant[i] - square(x), -2. * x);
  };

  const auto root = RootFinder::newton_raphson(func_and_deriv_lambda, guess,
                                               lower, upper, digits);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.NewtonRaphson.convergence_error.Double",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  test_throw_exception(
      []() {
        const size_t digits = 8;
        const double guess = 1.5;
        const double lower = 1.;
        const double upper = 2.;
        const auto func_and_deriv = [](double x) noexcept {
          return std::make_pair(2. - square(x), -2. * x);
        };
        RootFinder::newton_raphson(func_and_deriv, guess, lower, upper, digits,
                                   2);
      },
      convergence_error(
          "newton_raphson reached max iterations without converging"));
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.NewtonRaphson.convergence_error.DataVector",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  test_throw_exception(
      []() {
        const size_t digits = 8;
        const DataVector guess{1.6, 1.9, -1.6, -1.9};
        const DataVector lower{sqrt(2.), sqrt(2.), -2., -3.};
        const DataVector upper{2., 3., -sqrt(2.), -sqrt(2.)};
        const DataVector constant{2., 4., 2., 4.};
        const auto func_and_deriv = [&constant](const double x,
                                                const size_t i) noexcept {
          return std::make_pair(constant[i] - square(x), -2. * x);
        };
        RootFinder::newton_raphson(func_and_deriv, guess, lower, upper, digits,
                                   2);
      },
      convergence_error(
          "newton_raphson reached max iterations without converging"));
}
