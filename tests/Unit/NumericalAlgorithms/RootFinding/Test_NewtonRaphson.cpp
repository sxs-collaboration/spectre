// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"

namespace {
std::pair<double, double> func_and_deriv_free(double x) noexcept {
  return std::make_pair(2. - square(x), -2. * x);
}
struct FuncAndDeriv {
  std::pair<double, double> operator()(double x) const noexcept {
    return std::make_pair(2. - square(x), -2. * x);
  }
};

void test_simple() noexcept {
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

void test_bounds() noexcept {
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

void test_datavector() noexcept {
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

void test_convergence_error_double() noexcept {
  const size_t max_iterations = 2;
  const size_t digits = 8;
  const double guess = 1.5;
  const double lower = 1.;
  const double upper = 2.;
  boost::uintmax_t max_iters = max_iterations;

  const auto func_and_deriv = [](double x) noexcept {
    return std::make_pair(2. - square(x), -2. * x);
  };

  // Exception message is expected to print the "best" result, so here
  // we obtain that result directly with boost, as RootFinder::newton_raphson
  // would throw an exception.
  // clang-tidy: internal boost warning, can't fix it.
  const auto best_result =
      boost::math::tools::newton_raphson_iterate(  // NOLINT
          func_and_deriv, guess, lower, upper,
          std::round(std::log2(std::pow(10, digits))), max_iters);
  test_throw_exception(
      [&func_and_deriv, &guess, &lower, &upper, &max_iters]() {
        RootFinder::newton_raphson(func_and_deriv, guess, lower, upper, digits,
                                   max_iters);
      },
      convergence_error(MakeString{}
                        << "newton_raphson reached max iterations of "
                        << max_iterations
                        << " without converging. Best result is: "
                        << best_result << " with residual "
                        << func_and_deriv(best_result).first));
}

void test_convergence_error_datavector() noexcept {
  const size_t max_iterations = 2;
  const size_t digits = 8;
  const auto digits_binary = std::round(std::log2(std::pow(10, digits)));
  const DataVector lower{sqrt(2.), sqrt(2.), -2., -3.};
  const DataVector upper{2., 3., -sqrt(2.), -sqrt(2.)};
  const DataVector constant{2., 4., 2., 4.};

  const auto func_and_deriv = [&constant](const double x,
                                          const size_t i) noexcept {
    return std::make_pair(constant[i] - square(x), -2. * x);
  };

  // We test with different guesses, the difference being the location of the
  // element in the DataVector expected to throw the exception.
  const std::array<DataVector, 4> guess{{{1.6, 1.9, -1.6, -1.9},
                                         {sqrt(2.), 1.9, -1.6, -1.9},
                                         {sqrt(2.), 2.0, -1.6, -1.9},
                                         {sqrt(2.), 2.0, -sqrt(2.), -1.9}}};
  for (size_t i = 0; i < guess.size(); ++i) {
    const DataVector& current_guess = gsl::at(guess, i);
    // Exception message is expected to print the "best" result, so here
    // we obtain that result directly with boost, as RootFinder::newton_raphson
    // would throw an exception.
    DataVector best_result_vector{lower.size()};
    for (size_t s = 0; s < best_result_vector.size(); ++s) {
      boost::uintmax_t max_iters = max_iterations;
      // clang-tidy: internal boost warning, can't fix it.
      best_result_vector[s] =
          boost::math::tools::newton_raphson_iterate(  // NOLINT
              [&func_and_deriv, s ](double x) noexcept {
                return func_and_deriv(x, s);
              },
              current_guess[s], lower[s], upper[s], digits_binary, max_iters);
    }
    for (size_t s = 0; s < best_result_vector.size(); ++s) {
      boost::uintmax_t max_iters = max_iterations;
      test_throw_exception(
          [&func_and_deriv, &current_guess, &lower, &upper, &max_iters]() {
            RootFinder::newton_raphson(func_and_deriv, current_guess, lower,
                                       upper, digits, max_iters);
          },
          convergence_error(MakeString{}
                            << "newton_raphson reached max iterations of "
                            << max_iterations
                            << " without converging. Best result is: "
                            << best_result_vector[i] << " with residual "
                            << func_and_deriv(best_result_vector[i], i).first));
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.NewtonRaphson",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  test_simple();
  test_bounds();
  test_datavector();
  test_convergence_error_double();
  test_convergence_error_datavector();
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
}  // namespace
