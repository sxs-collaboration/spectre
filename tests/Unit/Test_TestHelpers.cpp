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
}

SPECTRE_TEST_CASE("Test.TestHelpers.Derivative", "[Unit]") {
  const std::array<double, 3> x{{1.2, -3.4, 1.3}};
  const double delta = 1.e-2;

  const auto func = [](const std::array<double, 3>& y) {
    return std::array<double, 3>{{sin(y[0]), cos(y[1]), exp(y[2])}};
  };
  const auto dfunc = [](const std::array<double, 3>& y) {
    return std::array<double, 3>{{cos(y[0]), -sin(y[1]), exp(y[2])}};
  };

  for(size_t i=0; i<3; ++i){
    CHECK(gsl::at(numerical_derivative(func, x, i, delta), i) ==
          approx(gsl::at(dfunc(x), i)));
  }
}
