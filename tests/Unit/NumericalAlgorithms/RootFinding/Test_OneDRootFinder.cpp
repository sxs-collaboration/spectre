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
                  "[Numerical][RootFinding][Unit]") {
  double abs_tol = 1e-15;
  double rel_tol = 1e-15;
  double upper = 2.0;
  double lower = 0.0;
  auto f_lambda = [](double x) { return 2.0 - x * x; };
  F f_functor;
  auto root_from_lambda =
      find_root_of_function(f_lambda, lower, upper, abs_tol, rel_tol);
  auto root_from_free =
      find_root_of_function(f_free, lower, upper, abs_tol, rel_tol);
  auto root_from_functor =
      find_root_of_function(f_functor, lower, upper, abs_tol, rel_tol);
  CHECK(std::abs(root_from_lambda - sqrt(2)) < abs_tol);
  CHECK(std::abs(root_from_lambda - sqrt(2)) / sqrt(2) < rel_tol);
  CHECK(root_from_free == root_from_lambda);
  CHECK(root_from_free == root_from_functor);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748RootSolver.Bounds",
                  "[Numerical][RootFinding][Unit]") {
  double abs_tol = 1e-15;
  double rel_tol = 1e-15;
  double upper = 2.0;
  double lower = sqrt(2.);
  auto f_lambda = [](double x) { return 2.0 - x * x; };

  auto root = find_root_of_function(f_lambda, lower, upper, abs_tol, rel_tol);
  CHECK(std::abs(root - sqrt(2)) < abs_tol);
  CHECK(std::abs(root - sqrt(2)) / sqrt(2) < rel_tol);

  lower = 0.;
  upper = sqrt(2.);

  root = find_root_of_function(f_lambda, lower, upper, abs_tol, rel_tol);
  CHECK(std::abs(root - sqrt(2)) < abs_tol);
  CHECK(std::abs(root - sqrt(2)) / sqrt(2) < rel_tol);
}

