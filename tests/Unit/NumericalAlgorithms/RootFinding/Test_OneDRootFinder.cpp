// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "NumericalAlgorithms/RootFinding/RootFinder.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
double f_free(double x) { return 2.0 - x * x; }
struct F {
  double operator()(double x) { return 2.0 - x * x; }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748RootSolver",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  const double upper = 2.0;
  const double lower = 0.0;
  const auto f_lambda = [](double x) { return 2.0 - x * x; };
  const F f_functor{};
  const auto root_from_lambda =
      find_root_of_function(f_lambda, lower, upper, abs_tol, rel_tol);
  const auto root_from_free =
      find_root_of_function(f_free, lower, upper, abs_tol, rel_tol);
  const auto root_from_functor =
      find_root_of_function(f_functor, lower, upper, abs_tol, rel_tol);
  CHECK(std::abs(root_from_lambda - sqrt(2)) < abs_tol);
  CHECK(std::abs(root_from_lambda - sqrt(2)) / sqrt(2) < rel_tol);
  CHECK(root_from_free == root_from_lambda);
  CHECK(root_from_free == root_from_functor);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748RootSolver.Bounds",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  /// [double_root_find]
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  double upper = 2.0;
  double lower = sqrt(2.);
  const auto f_lambda = [](double x) { return 2.0 - x * x; };

  auto root = find_root_of_function(f_lambda, lower, upper, abs_tol, rel_tol);
  /// [double_root_find]

  CHECK(std::abs(root - sqrt(2)) < abs_tol);
  CHECK(std::abs(root - sqrt(2)) / sqrt(2) < rel_tol);

  lower = 0.;
  upper = sqrt(2.);

  root = find_root_of_function(f_lambda, lower, upper, abs_tol, rel_tol);
  CHECK(std::abs(root - sqrt(2)) < abs_tol);
  CHECK(std::abs(root - sqrt(2)) / sqrt(2) < rel_tol);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748RootSolver.DataVector",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  /// [datavector_root_find]
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  const DataVector upper{2.0, 3.0, -sqrt(2.0), -sqrt(2.0)};
  const DataVector lower{sqrt(2.), sqrt(2.0), -2.0, -3.0};

  const DataVector constant{2.0, 4.0, 2.0, 4.0};
  const auto f_lambda = [&constant](const double x, const size_t i) noexcept {
    return constant[i] - x * x;
  };

  const auto root =
      find_root_of_function(f_lambda, lower, upper, abs_tol, rel_tol);
  /// [datavector_root_find]

  CHECK(std::abs(root[0] - sqrt(2.0)) < abs_tol);
  CHECK(std::abs(root[0] - sqrt(2.0)) / sqrt(2.0) < rel_tol);
  CHECK(std::abs(root[1] - 2.0) < abs_tol);
  CHECK(std::abs(root[1] - 2.0) / 2.0 < rel_tol);
  CHECK(std::abs(root[2] + sqrt(2.0)) < abs_tol);
  CHECK(std::abs(root[2] + sqrt(2.0)) / sqrt(2.0) < rel_tol);
  CHECK(std::abs(root[3] + 2.0) < abs_tol);
  CHECK(std::abs(root[3] + 2.0) / 2.0 < rel_tol);
}
