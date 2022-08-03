// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"

namespace {
double f_free(double x) { return 2.0 - square(x); }
struct F {
  template <typename T>
  T operator()(const T& x) const {
    return 2.0 - square(x);
  }
};

void test_simple() {
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

void test_bounds() {
  // [double_root_find]
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  const double upper = 2.0;
  const double lower = sqrt(2.0) - abs_tol;  // bracket surrounds root
  const auto f_lambda = [](double x) { return 2.0 - square(x); };

  auto root = RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
  // [double_root_find]

  CHECK(std::abs(root - sqrt(2.0)) < abs_tol);
  CHECK(std::abs(root - sqrt(2.0)) / sqrt(2.0) < rel_tol);

  // Test the overload where function values are supplied at lower
  // and upper bounds.
  auto root2 = RootFinder::toms748(f_lambda, lower, upper, f_lambda(lower),
                                   f_lambda(upper), abs_tol, rel_tol);
  CHECK(std::abs(root2 - sqrt(2.0)) < abs_tol);
  CHECK(std::abs(root2 - sqrt(2.0)) / sqrt(2.0) < rel_tol);

  // Check that the other tight-but-correct bracket works
  CHECK(RootFinder::toms748(f_lambda, 0.0, sqrt(2.0) + abs_tol, abs_tol,
                            rel_tol) == approx(root));

  // Check that exception is thrown for various bad bracket possibilities
  CHECK_THROWS_WITH(
      RootFinder::toms748(f_lambda, 0.0, sqrt(2.0) - abs_tol, abs_tol, rel_tol),
      Catch::Matchers::ContainsSubstring("Root not bracketed"));

  CHECK_THROWS_WITH(
      RootFinder::toms748(f_lambda, sqrt(2.0) + abs_tol, 2.0, abs_tol, rel_tol),
      Catch::Matchers::ContainsSubstring("Root not bracketed"));

  CHECK_THROWS_WITH(RootFinder::toms748(f_lambda, -1.0, 1.0, abs_tol, rel_tol),
                    Catch::Matchers::ContainsSubstring("Root not bracketed"));
}

void test_datavector() {
  // [datavector_root_find]
  const double abs_tol = 1e-15;
  const double rel_tol = 1e-15;
  const DataVector upper{2.0, 3.0, -sqrt(2.0) + abs_tol, -sqrt(2.0)};
  const DataVector lower{sqrt(2.0) - abs_tol, sqrt(2.0), -2.0, -3.0};

  const DataVector constant{2.0, 4.0, 2.0, 4.0};
  const auto f_lambda = [&constant](const auto x, const size_t i) {
    if constexpr (simd::is_batch<std::decay_t<decltype(x)>>::value) {
      return simd::load_unaligned(&(constant[i])) - square(x);
    } else {
      return constant[i] - square(x);
    }
  };

  const auto root_no_function_values =
      RootFinder::toms748<true>(f_lambda, lower, upper, abs_tol, rel_tol);
  // [datavector_root_find]

  auto check_root = [&abs_tol, &rel_tol](const DataVector& root) {
    CHECK(std::abs(root[0] - sqrt(2.0)) < abs_tol);
    CHECK(std::abs(root[0] - sqrt(2.0)) / sqrt(2.0) < rel_tol);
    CHECK(std::abs(root[1] - 2.0) < abs_tol);
    CHECK(std::abs(root[1] - 2.0) / 2.0 < rel_tol);
    CHECK(std::abs(root[2] + sqrt(2.0)) < abs_tol);
    CHECK(std::abs(root[2] + sqrt(2.0)) / sqrt(2.0) < rel_tol);
    CHECK(std::abs(root[3] + 2.0) < abs_tol);
    CHECK(std::abs(root[3] + 2.0) / 2.0 < rel_tol);
  };
  check_root(root_no_function_values);

  // Test the version of toms748 where function values are supplied
  // at lower and upper bounds.
  const auto generate_function_values = [&f_lambda](const DataVector& x) {
    DataVector f(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      f[i] = f_lambda(x[i], i);
    }
    return f;
  };
  const auto root_function_values = RootFinder::toms748<true>(
      f_lambda, lower, upper, generate_function_values(lower),
      generate_function_values(upper), abs_tol, rel_tol);
  check_root(root_function_values);
}

void test_convergence_error_double() {
  CHECK_THROWS_AS(
      []() {
        const double abs_tol = 1e-15;
        const double rel_tol = 1e-15;
        const double upper = 2.0;
        const double lower = 0.0;
        const auto f = [](double x) { return 2.0 - square(x); };
        RootFinder::toms748(f, lower, upper, abs_tol, rel_tol, 2);
      }(),
      convergence_error);
}

void test_convergence_error_datavector() {
  CHECK_THROWS_AS(
      ([]() {
        const double abs_tol = 1e-15;
        const double rel_tol = 1e-15;
        const DataVector upper{2.0, 3.0, -sqrt(2.0) + abs_tol, -sqrt(2.0)};
        const DataVector lower{sqrt(2.0) - abs_tol, sqrt(2.0), -2.0, -3.0};
        const DataVector constant{2.0, 4.0, 2.0, 4.0};
        const auto f = [&constant](const auto x, const size_t i) {
          if constexpr (simd::is_batch<std::decay_t<decltype(x)>>::value) {
            return simd::load_unaligned(&constant[i]) - square(x);
          } else {
            return constant[i] - square(x);
          }
        };
        RootFinder::toms748(f, lower, upper, abs_tol, rel_tol, 2);
      }()),
      convergence_error);
}

void benchmark_root_find(const bool enable) {
  if (not enable) {
    return;
  }

#ifdef SPECTRE_USE_XSIMD
  using SimdType = simd::batch<double>;
  const auto benchmark = [](const int bound_type,
                            Catch::Benchmark::Chronometer& meter) {
    const auto f = [](const auto& x) {
      using T = std::decay_t<decltype(x)>;
      return simd::fms(x, simd::fma(x, (x - 6.0), T(11.0)), T(6.0));
    };
    const SimdType lower_bound(1.5);
    SimdType upper_bound{};
    if (bound_type == 0) {
      upper_bound = SimdType(2.5 + 1.0e-100) + 1.0e-8;
    } else if (bound_type == 1) {
      ERROR("Need to adjust");
      // upper_bound = SimdType(2.1 + 1.0e-10, 2.1 + 1.0e-10, 2.1 + 1.0e-10) +
      //               1.0e-8;
    } else if (bound_type == 2) {
      ERROR("Need to adjust");
      // upper_bound = SimdType(2.1 + 1.0e-10, 2.2 + 1.0e-10, 2.3 + 1.0e-10,
      //                        2.4 + 1.0e-10, 2.5, 2.6, 2.7, 2.8) +
      //               1.0e-8;
    } else if (bound_type == 3) {
      ERROR("Need to adjust");
      // const double base = 2.8 + 1.0e-8;
      // upper_bound = SimdType(base + 1.0e-10, base + 2.0e-10, base + 3.0e-10,
      //                        base + 4.0e-10, base + 5.0e-10, base + 6.0e-10,
      //                        base + 7.0e-10, base + 8.0e-10);
    }
    const auto f_at_lower_bound = f(lower_bound);
    const auto f_at_upper_bound = f(upper_bound);
    meter.measure([&f, &lower_bound, &upper_bound, &f_at_lower_bound,
                   &f_at_upper_bound]() {
      return RootFinder::toms748(f, lower_bound, upper_bound, f_at_lower_bound,
                                 f_at_upper_bound, 1.0e-14, 1.0e-12);
    });
  };

  for (size_t i = 0; i < 4; ++i) {
    // Can check other bound types, but those are currently hard-coded to
    // specific vector widths, which won't let the code compile generically.
    // The general support is left in case someone wants to look at this again.
    BENCHMARK_ADVANCED(std::string("advanced ") + std::to_string(i))
    (Catch::Benchmark::Chronometer meter) { benchmark(0, meter); };
  }
#endif // SPECTRE_USE_XSIMD
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.TOMS748",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  test_simple();
  test_bounds();
  test_datavector();
  test_convergence_error_double();
  test_convergence_error_datavector();
  benchmark_root_find(false);

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        const double abs_tol = 1e-15;
        const double rel_tol = 0.5 * std::numeric_limits<double>::epsilon();
        const DataVector upper{2.0, 3.0, -sqrt(2.0) + abs_tol, -sqrt(2.0)};
        const DataVector lower{sqrt(2.0) - abs_tol, sqrt(2.0), -2.0, -3.0};

        const DataVector constant{2.0, 4.0, 2.0, 4.0};
        const auto f_lambda = [&constant](const auto x, const size_t i) {
          if constexpr (simd::is_batch<std::decay_t<decltype(x)>>::value) {
            return simd::load_unaligned(&constant[i]) - square(x);
          } else {
            return constant[i] - square(x);
          }
        };

        RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
      }()),
      Catch::Matchers::ContainsSubstring(
          "The relative tolerance is too small."));
  CHECK_THROWS_WITH(
      ([]() {
        const double abs_tol = 1e-15;
        const double rel_tol = 0.5 * std::numeric_limits<double>::epsilon();
        double upper = 2.0;
        double lower = sqrt(2.0) - abs_tol;
        const auto f_lambda = [](double x) { return 2.0 - square(x); };

        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(f_lambda, lower, upper, abs_tol, rel_tol);
      }()),
      Catch::Matchers::ContainsSubstring(
          "The relative tolerance is too small."));
#endif
}
}  // namespace
