// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <tuple>

#include "Utilities/CartesianProduct.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
void test_product_of_arrays() {
  const auto result =
      cartesian_product(make_array(true, false), make_array(1, 2, 3));
  const std::array<std::tuple<bool, int>, 6> expected{
      {{true, 1}, {true, 2}, {true, 3}, {false, 1}, {false, 2}, {false, 3}}};
  CHECK(result == expected);
}

void test_product_of_containers() {
  std::array bools{true, false};
  std::vector ints{1, 2, 3};
  std::set doubles{-1., 0.5, 3.0, 7.25};
  const auto result = cartesian_product(bools, ints, doubles);
  const std::vector<std::tuple<bool, int, double>> expected{
      {true, 1, -1.},  {true, 1, 0.5},  {true, 1, 3.0},  {true, 1, 7.25},
      {true, 2, -1.},  {true, 2, 0.5},  {true, 2, 3.0},  {true, 2, 7.25},
      {true, 3, -1.},  {true, 3, 0.5},  {true, 3, 3.0},  {true, 3, 7.25},
      {false, 1, -1.}, {false, 1, 0.5}, {false, 1, 3.0}, {false, 1, 7.25},
      {false, 2, -1.}, {false, 2, 0.5}, {false, 2, 3.0}, {false, 2, 7.25},
      {false, 3, -1.}, {false, 3, 0.5}, {false, 3, 3.0}, {false, 3, 7.25}};
  CHECK(result == expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.CartesianProduct", "[Unit][Utilities]") {
  test_product_of_arrays();
  test_product_of_containers();
}
