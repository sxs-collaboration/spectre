// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
double f_free(double x) { return 2.0 - square(x); }
struct F {
  double operator()(double x) { return 2.0 - square(x); }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  const double upper = 2.0;
  const double lower = 0.0;
  const auto f_lambda = [](double x) { return 2.0 - square(x); };
  const F f_functor{};
  const auto root_from_lambda =
      RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
  const auto root_from_free =
      RootFinder::toms748(f_free, lower, upper, abs_tol, rel_tol);
  const auto root_from_functor =
      RootFinder::toms748(f_functor, lower, upper, abs_tol, rel_tol);
  CHECK(std::abs(root_from_lambda - sqrt(2)) < abs_tol);
  CHECK(std::abs(root_from_lambda - sqrt(2)) / sqrt(2) < rel_tol);
  CHECK(root_from_free == root_from_lambda);
  CHECK(root_from_free == root_from_functor);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748.Bounds",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  /// [double_root_find]
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  double upper = 2.0;
  double lower = sqrt(2.0);
  const auto f_lambda = [](double x) { return 2.0 - square(x); };

  auto root = RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
  /// [double_root_find]

  CHECK(std::abs(root - sqrt(2)) < abs_tol);
  CHECK(std::abs(root - sqrt(2)) / sqrt(2) < rel_tol);

  lower = 0.;
  upper = sqrt(2.);

  root = RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
  CHECK(std::abs(root - sqrt(2)) < abs_tol);
  CHECK(std::abs(root - sqrt(2)) / sqrt(2) < rel_tol);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748.DataVector",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  /// [datavector_root_find]
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  const DataVector upper{2.0, 3.0, -sqrt(2.0), -sqrt(2.0)};
  const DataVector lower{sqrt(2.), sqrt(2.0), -2.0, -3.0};

  const DataVector constant{2.0, 4.0, 2.0, 4.0};
  const auto f_lambda = [&constant](const double x, const size_t i) noexcept {
    return constant[i] - square(x);
  };

  const auto root =
      RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
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

// [[OutputRegex, The relative tolerance is too small.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.TOMS748.RelativeTol.DataVector",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const double abs_tol = 1e-15;
  const double rel_tol = 0.5 * std::numeric_limits<double>::epsilon();
  const DataVector upper{2.0, 3.0, -sqrt(2.0), -sqrt(2.0)};
  const DataVector lower{sqrt(2.0), sqrt(2.0), -2.0, -3.0};

  const DataVector constant{2.0, 4.0, 2.0, 4.0};
  const auto f_lambda = [&constant](const double x, const size_t i) noexcept {
    return constant[i] - square(x);
  };

  RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The relative tolerance is too small.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.RootFinding.TOMS748.RelativeTol.Double",
    "[NumericalAlgorithms][RootFinding][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const double abs_tol = 1e-15;
  const double rel_tol = 0.5 * std::numeric_limits<double>::epsilon();
  double upper = 2.0;
  double lower = sqrt(2.0);
  const auto f_lambda = [](double x) { return 2.0 - square(x); };

  RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
