// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <boost/algorithm/string/predicate.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

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
  CHECK(std::abs(root_from_lambda - sqrt(2.0)) < abs_tol);
  CHECK(std::abs(root_from_lambda - sqrt(2.0)) / sqrt(2.0) < rel_tol);
  CHECK(root_from_free == root_from_lambda);
  CHECK(root_from_free == root_from_functor);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748.Bounds",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  /// [double_root_find]
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  const double upper = 2.0;
  const double lower = sqrt(2.0) - abs_tol;  // bracket surrounds root
  const auto f_lambda = [](double x) { return 2.0 - square(x); };

  auto root = RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
  /// [double_root_find]

  CHECK(std::abs(root - sqrt(2.0)) < abs_tol);
  CHECK(std::abs(root - sqrt(2.0)) / sqrt(2.0) < rel_tol);

  // Check that the other tight-but-correct bracket works
  CHECK(RootFinder::toms748(f_lambda, 0.0, sqrt(2.0) + abs_tol, abs_tol,
                            rel_tol) == approx(root));

  // Check that exception is thrown for various bad bracket possibilities
  const auto test_bad_bracket_exception = [&f_lambda, &abs_tol, &rel_tol](
                                              const double local_lower,
                                              const double local_upper,
                                              const std::string& msg) {
    try {
      RootFinder::toms748(f_lambda, local_lower, local_upper, abs_tol, rel_tol);
      INFO(msg);
      CHECK(false);
    } catch (std::domain_error& e) {
      const std::string expected =
          "Error in function boost::math::tools::toms748_solve<double>: "
          "Parameters a and b do not bracket the root:";
      CAPTURE(e.what());
      CHECK(boost::algorithm::starts_with(e.what(), expected));
    } catch (...) {
      CHECK(false);
    }
  };

  test_bad_bracket_exception(
      0.0, sqrt(2.0) - abs_tol,
      "Expected root finder to fail because upper bound is too tight");
  test_bad_bracket_exception(
      sqrt(2.0) + abs_tol, upper,
      "Expected root finder to fail because lower bound is too tight");
  test_bad_bracket_exception(
      -1.0, 1.0, "Expected root finder to fail because root is not bracketed");
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748.DataVector",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  /// [datavector_root_find]
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  const DataVector upper{2.0, 3.0, -sqrt(2.0) + abs_tol, -sqrt(2.0)};
  const DataVector lower{sqrt(2.0) - abs_tol, sqrt(2.0), -2.0, -3.0};

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
  const DataVector upper{2.0, 3.0, -sqrt(2.0) + abs_tol, -sqrt(2.0)};
  const DataVector lower{sqrt(2.0) - abs_tol, sqrt(2.0), -2.0, -3.0};

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
  double lower = sqrt(2.0) - abs_tol;
  const auto f_lambda = [](double x) { return 2.0 - square(x); };

  RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
