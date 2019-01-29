// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.TestHelpers", "[Unit]") {
  std::vector<double> vector{0, 1, 2, 3};
  test_iterators(vector);
  test_reverse_iterators(vector);

  std::set<double> set;
  set.insert(0);
  set.insert(1);
  set.insert(2);
  set.insert(3);
  test_iterators(set);
  test_reverse_iterators(set);

  std::unordered_set<int> u_set;
  u_set.insert(3);
  u_set.insert(2);
  u_set.insert(1);
  u_set.insert(0);
  test_iterators(u_set);

  Approx larger_approx =
      Approx::custom().epsilon(std::numeric_limits<double>::epsilon() * 1.e4);

  const std::vector<double> vec_a{1., 2., 3.5};
  CHECK_ITERABLE_APPROX(vec_a, vec_a);
  auto vec_b = vec_a;
  vec_b[1] += 1e-15;
  CHECK(vec_a != vec_b);
  CHECK_ITERABLE_APPROX(vec_a, vec_b);
  vec_b[1] += 1e-12;
  CHECK_ITERABLE_CUSTOM_APPROX(vec_a, vec_b, larger_approx);

  const std::vector<std::complex<double>> complex_vector{
      std::complex<double>(1.0, 1.5), std::complex<double>(2.0, 2.5),
      std::complex<double>(3.0, 3.5)};
  CHECK_ITERABLE_APPROX(complex_vector, complex_vector);
  auto perturbed_complex_vector = complex_vector;
  perturbed_complex_vector[1] += 1e-15;
  CHECK(complex_vector != perturbed_complex_vector);
  CHECK_ITERABLE_APPROX(complex_vector, perturbed_complex_vector);
  perturbed_complex_vector[1] += 1e-12;
  CHECK_ITERABLE_CUSTOM_APPROX(complex_vector, perturbed_complex_vector,
                               larger_approx);

  const std::vector<std::map<int, double>> vecmap_a{
      {{1, 1.}, {2, 2.}}, {{1, 1.23}, {3, 4.56}, {5, 7.89}}};
  CHECK_ITERABLE_APPROX(vecmap_a, vecmap_a);
  auto vecmap_b = vecmap_a;
  vecmap_b[1][1] += 1e-15;
  CHECK(vecmap_a != vecmap_b);
  CHECK_ITERABLE_APPROX(vecmap_a, vecmap_b);
  vecmap_b[1][1] += 1e-12;
  CHECK_ITERABLE_CUSTOM_APPROX(vecmap_a, vecmap_b, larger_approx);

  // Check that CAPTURE_PRECISE accepts an STL type (we cannot test
  // the output because that is only produced on a Catch failure,
  // which would fail the test.
  {
    CAPTURE_PRECISE((std::array<double, 1>{{1.5}}));
  }

  // Check that CHECK_ITERABLE_APPROX works on various containers
  {
    const std::set<int> a{1, 2, 3};
    CHECK_ITERABLE_APPROX(a, a);
  }
  {
    // Iteration order is unspecified, but we create the containers
    // differently so the order might differ between them.
    const std::unordered_set<int> a{1, 2, 3};
    const std::unordered_set<int> b{3, 2, 1};
    CHECK_ITERABLE_APPROX(a, b);
  }
  {
    // Iteration order is unspecified, but we create the containers
    // differently so the order might differ between them.
    const std::unordered_map<int, int> a{{1, 2}, {2, 3}, {3, 4}};
    const std::unordered_map<int, int> b{{3, 4}, {2, 3}, {1, 2}};
    CHECK_ITERABLE_APPROX(a, b);
  }
}

SPECTRE_TEST_CASE("Unit.TestHelpers.Derivative", "[Unit]") {
  {  // 3D Test
    const std::array<double, 3> x{{1.2, -3.4, 1.3}};
    const double delta = 1.e-2;

    const auto func = [](const std::array<double, 3>& y) {
      return std::array<double, 3>{{sin(y[0]), cos(y[1]), exp(y[2])}};
    };
    const auto dfunc = [](const std::array<double, 3>& y) {
      return std::array<double, 3>{{cos(y[0]), -sin(y[1]), exp(y[2])}};
    };

    for (size_t i = 0; i < 3; ++i) {
      CHECK(gsl::at(numerical_derivative(func, x, i, delta), i) ==
            approx(gsl::at(dfunc(x), i)));
    }
  }
  {  // 2D Test
    const std::array<double, 2> x{{1.2, -2.4}};
    const double delta = 1.e-2;

    const auto func = [](const std::array<double, 2>& y) {
      return std::array<double, 2>{{sin(y[0]), cos(y[1])}};
    };
    const auto dfunc = [](const std::array<double, 2>& y) {
      return std::array<double, 2>{{cos(y[0]), -sin(y[1])}};
    };

    for (size_t i = 0; i < 2; ++i) {
      CHECK(gsl::at(numerical_derivative(func, x, i, delta), i) ==
            approx(gsl::at(dfunc(x), i)));
    }
  }
}

SPECTRE_TEST_CASE("Unit.TestHelpers.MAKE_GENERATOR", "[Unit]") {
  MAKE_GENERATOR(gen1);
  MAKE_GENERATOR(gen2);
  // This will fail randomly every 2**32 runs.  That is probably OK.
  CHECK(gen1() != gen2());

  MAKE_GENERATOR(seeded_gen, 12345);
  CHECK(seeded_gen() == 3992670690);
}
