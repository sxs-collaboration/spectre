// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"

namespace {
std::pair<double, double> func_and_deriv_free(double x) {
  return std::make_pair(2. - square(x), -2. * x);
}
struct FuncAndDeriv {
  std::pair<double, double> operator()(double x) const {
    return std::make_pair(2. - square(x), -2. * x);
  }
};

void test_simple() {
  // [double_newton_raphson_root_find]
  const double residual_tolerance = 1.0e-14;
  const double step_absolute_tolerance = 1.0e-8;
  const double step_relative_tolerance = 0.0;
  const double correct = sqrt(2.);
  const double guess = 1.5;
  const double lower = 1.;
  const double upper = 2.;
  const auto func_and_deriv_lambda = [](double x) {
    return std::make_pair(2. - square(x), -2. * x);
  };
  const auto root_from_lambda = RootFinder::newton_raphson(
      func_and_deriv_lambda, guess, lower, upper, residual_tolerance,
      step_absolute_tolerance, step_relative_tolerance);
  // [double_newton_raphson_root_find]
  const auto root_from_free = RootFinder::newton_raphson(
      func_and_deriv_free, guess, lower, upper, residual_tolerance,
      step_absolute_tolerance, step_relative_tolerance);
  const FuncAndDeriv func_and_deriv_functor{};
  const auto root_from_functor = RootFinder::newton_raphson(
      func_and_deriv_functor, guess, lower, upper, residual_tolerance,
      step_absolute_tolerance, step_relative_tolerance);
  CHECK(std::abs(root_from_lambda - correct) <= step_absolute_tolerance);
  CHECK(root_from_free == root_from_lambda);
  CHECK(root_from_free == root_from_functor);
}

void test_bounds() {
  const double residual_tolerance = 1.0e-14;
  const double step_absolute_tolerance = 1.0e-8;
  const double step_relative_tolerance = 0.0;
  const double guess = 1.5;
  double upper = 2.;
  double lower = sqrt(2.);
  const auto func_and_deriv_lambda = [](double x) {
    return std::make_pair(2. - square(x), -2. * x);
  };

  auto root = RootFinder::newton_raphson(
      func_and_deriv_lambda, guess, lower, upper, residual_tolerance,
      step_absolute_tolerance, step_relative_tolerance);
  const double correct = sqrt(2.);

  CHECK(std::abs(root - correct) < step_absolute_tolerance);

  lower = 0.;
  upper = sqrt(2.);

  root = RootFinder::newton_raphson(func_and_deriv_lambda, guess, lower, upper,
                                    residual_tolerance, step_absolute_tolerance,
                                    step_relative_tolerance);
  CHECK(std::abs(root - correct) < step_absolute_tolerance);
}

void test_datavector() {
  // [datavector_newton_raphson_root_find]
  const double residual_tolerance = 1.0e-14;
  const double step_absolute_tolerance = 1.0e-8;
  const double step_relative_tolerance = 0.0;
  const DataVector guess{1.6, 1.9, -1.6, -1.9};
  const DataVector lower{sqrt(2.), sqrt(2.), -2., -3.};
  const DataVector upper{2., 3., -sqrt(2.), -sqrt(2.)};
  const DataVector constant{2., 4., 2., 4.};

  const auto func_and_deriv_lambda = [&constant](const double x,
                                                 const size_t i) {
    return std::make_pair(constant[i] - square(x), -2. * x);
  };

  const auto root = RootFinder::newton_raphson(
      func_and_deriv_lambda, guess, lower, upper, residual_tolerance,
      step_absolute_tolerance, step_relative_tolerance);
  // [datavector_newton_raphson_root_find]

  const DataVector correct{sqrt(2.), 2., -sqrt(2.), -2.};

  for (size_t i = 0; i < guess.size(); i++) {
    CHECK(std::abs(root[i] - correct[i]) < step_absolute_tolerance);
  }
}

void test_convergence_error_double() {
  const size_t max_iterations = 2;
  const double residual_tolerance = 1.0e-14;
  const double step_absolute_tolerance = 1.0e-8;
  const double step_relative_tolerance = 0.0;
  const double guess = 1.5;
  const double lower = 1.;
  const double upper = 2.;

  const auto func_and_deriv = [](double x) {
    return std::make_pair(2. - square(x), -2. * x);
  };

  CHECK_THROWS_AS(
      RootFinder::newton_raphson(func_and_deriv, guess, lower, upper,
                                 residual_tolerance, step_absolute_tolerance,
                                 step_relative_tolerance, max_iterations),
      convergence_error);
}

void test_bad_guess() {
  // We switched away from the Boost implementation because it failed in
  // some cases where the guess was bad.  This is one such case.
  {
    const auto quartic = [](const double x) {
      return std::make_pair((x * x - 1.0) * (x * x + 0.1),
                            2.0 * x * ((x * x - 1.0) + (x * x + 0.1)));
    };

    const double result =
        RootFinder::newton_raphson(quartic, 0.5, 0.1, 1.1, 0.0, 1.0e-5, 0.0);
    CHECK(std::abs(result - 1.0) < 1.0e-5);
  }

  // Verify that a really ugly function works.
  {
    const auto ugly = [](const double x) {
      const double a = x * x - 20.0;
      const double da = 2.0 * x;
      const double b = sin(6.0 * x) + 1.1;
      const double db = 6.0 * cos(6.0 * x);
      return std::make_pair(a * b, a * db + da * b);
    };
    const double correct = sqrt(20.0);
    const double tolerance = 1.0e-6;
    Approx ugly_approx = Approx::custom().epsilon(tolerance).scale(1.0);

    const double trials = 101.0;
    for (double i = 0.0; i <= trials; ++i) {
      CHECK(RootFinder::newton_raphson(ugly, i / trials, 0.0, 10.0, 0.0,
                                       tolerance, 0.0) == ugly_approx(correct));
    }
  }
}

void test_convergence_reasons() {
  double correct = 0.5;
  const auto f = [&correct](const double x) {
    const double x_off = x - correct + sqrt(2.0);
    return std::make_pair(x_off * x_off - 2.0, 2.0 * x_off);
  };

  // Convergence: Absolute tolerance
  {
    const double root_from_small_tolerance =
        RootFinder::newton_raphson(f, 1.0, 0.0, 5.0, 0.0, 1.0e-10, 0.0);
    const double root_from_large_tolerance =
        RootFinder::newton_raphson(f, 1.0, 0.0, 5.0, 0.0, 1.0e-1, 0.0);
    CHECK(std::abs(root_from_small_tolerance - correct) < 1.0e-10);
    CHECK(std::abs(root_from_large_tolerance - correct) < 1.0e-1);
    // The convergence tolerances checked are usually good estimates
    // of the error in the /previous/ step, so the actual returned
    // value is often much more accurate than required.  Check that
    // the smaller tolerance gave a more accurate result, without
    // worrying about exactly how accurate.
    CHECK(std::abs(root_from_small_tolerance - correct) <
          std::abs(root_from_large_tolerance - correct));
  }

  // Convergence: Relative tolerance (and comparing to absolute)
  {
    correct = 1.0e-2;
    const double large_absolute =
        RootFinder::newton_raphson(f, 0.5, -1.0, 1.0, 0.0, 1.0e-1, 0.0);
    const double small_absolute =
        RootFinder::newton_raphson(f, 0.5, -1.0, 1.0, 0.0, 1.0e-6, 0.0);
    const double large_relative =
        RootFinder::newton_raphson(f, 0.5, -1.0, 1.0, 0.0, 0.0, 1.0e-1);
    const double small_relative =
        RootFinder::newton_raphson(f, 0.5, -1.0, 1.0, 0.0, 0.0, 1.0e-10);
    CHECK(std::abs(large_absolute - correct) < 1.0e-1);
    CHECK(std::abs(small_absolute - correct) < 1.0e-6);
    CHECK(std::abs(large_relative - correct) < 1.0e-3);
    CHECK(std::abs(small_relative - correct) < 1.0e-12);
    // The convergence tolerances checked are usually good estimates
    // of the error in the /previous/ step, so the actual returned
    // value is often much more accurate than required.  Check that
    // the smaller tolerance gave a more accurate result, without
    // worrying about exactly how accurate.
    CHECK(std::abs(small_absolute - correct) <
          std::abs(large_absolute - correct));
    CHECK(std::abs(small_relative - correct) <
          std::abs(large_relative - correct));
  }

  // Residual
  {
    correct = 1.0;
    const double large_residual =
        RootFinder::newton_raphson(f, 0.5, 0.0, 5.0, 1.0e-1, 0.0, 0.0);
    const double small_residual =
        RootFinder::newton_raphson(f, 0.5, 0.0, 5.0, 1.0e-8, 0.0, 0.0);
    CHECK(std::abs(f(large_residual).first) < 1.0e-1);
    CHECK(std::abs(f(large_residual).first) > 1.0e-8);
    CHECK(std::abs(f(small_residual).first) < 1.0e-8);
  }
}

void test_convergence_rate() {
  int call_count = 0;
  const auto smooth_f = [&call_count](const double x) {
    ++call_count;
    return std::make_pair(x * x - 2.0, 2.0 * x);
  };
  const double correct = sqrt(2.0);

  // Find the correction to counts from assertion checks to make sure
  // the test gives the same behavior in both build modes.

  // Finding the root with a huge residual tolerance should only
  // require one evaluation.
  call_count = 0;
  RootFinder::newton_raphson(smooth_f, 2.0, 0.0, 5.0, 1000.0, 0.0, 0.0);
  const int assertion_correction = call_count - 1;
#ifdef SPECTRE_DEBUG
  CHECK(assertion_correction != 0);
#else   // SPECTRE_DEBUG
  CHECK(assertion_correction == 0);
#endif  // SPECTRE_DEBUG

  call_count = 0;
  const double inaccurate =
      RootFinder::newton_raphson(smooth_f, 2.0, 0.0, 5.0, 1e-3, 0.0, 0.0);
  const int inaccurate_calls = call_count - assertion_correction;

  call_count = 0;
  const double accurate =
      RootFinder::newton_raphson(smooth_f, 2.0, 0.0, 5.0, 1.0e-8, 0.0, 0.0);
  const int accurate_calls = call_count - assertion_correction;
  CHECK(accurate_calls > inaccurate_calls);
  // Try to avoid being roundoff-dominated.
  REQUIRE(std::abs(accurate - correct) > 1.0e-14);

  // error_{n+1} ~ error_n^p  with  p ~ 2
  CHECK(pow(log(std::abs(accurate - correct)) /
                log(std::abs(inaccurate - correct)),
            1.0 / (accurate_calls - inaccurate_calls)) ==
        Approx::custom().margin(0.1)(2.0));
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.NewtonRaphson",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  test_simple();
  test_bounds();
  test_datavector();
  test_convergence_error_double();
  test_bad_guess();
  test_convergence_reasons();
  test_convergence_rate();
}
}  // namespace
