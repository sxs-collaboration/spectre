// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Test.TestHelpers", "[Unit]") {
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

  const std::vector<std::map<int, double>> vecmap_a{
      {{1, 1.}, {2, 2.}}, {{1, 1.23}, {3, 4.56}, {5, 7.89}}};
  CHECK_ITERABLE_APPROX(vecmap_a, vecmap_a);
  auto vecmap_b = vecmap_a;
  vecmap_b[1][1] += 1e-15;
  CHECK(vecmap_a != vecmap_b);
  CHECK_ITERABLE_APPROX(vecmap_a, vecmap_b);
  vecmap_b[1][1] += 1e-12;
  CHECK_ITERABLE_CUSTOM_APPROX(vecmap_a, vecmap_b, larger_approx);
}

SPECTRE_TEST_CASE("Test.TestHelpers.Derivative", "[Unit]") {
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
