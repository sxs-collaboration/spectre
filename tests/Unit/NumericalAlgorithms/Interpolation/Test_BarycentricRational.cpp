// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "NumericalAlgorithms/Interpolation/BarycentricRational.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <class F>
void test_barycentric_rational(const F& function, const double lower_bound,
                               const double upper_bound, const size_t size,
                               const size_t order) noexcept {
  std::vector<double> x_values(size), y_values(size);
  const double delta_x = (upper_bound - lower_bound) / size;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dis(0.0, delta_x);
  for (size_t i = 0; i < size; ++i) {
    x_values[i] = lower_bound + i * delta_x + dis(gen);
    y_values[i] = function(x_values[i]);
  }

  intrp::BarycentricRational interpolant{x_values, y_values, order};

  const auto deserialized_interpolant = serialize_and_deserialize(interpolant);
  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  for (size_t i = 0; i < 10 * size; ++i) {
    const double x_value = lower_bound + i * delta_x * 0.1 + 0.1 * dis(gen);
    CAPTURE(x_value);
    CHECK_ITERABLE_CUSTOM_APPROX(function(x_value), interpolant(x_value),
                                 custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        function(x_value), deserialized_interpolant(x_value), custom_approx);
  }
  CHECK(order == interpolant.order());
  CHECK(order == deserialized_interpolant.order());
}

void single_call(const size_t number_of_points, const size_t order,
                 const size_t expo) {
  INFO("Polynomial degree := " << expo);
  test_barycentric_rational(
      [expo](const auto& x) noexcept {
        auto result = x;
        for (size_t j = 2; j < expo; ++j) {
          result += pow(x, j);
        }
        return result;
      },
      -1.0, 2.3, number_of_points, order);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.BarycentricRational",
                  "[Unit][NumericalAlgorithms]") {
  for (size_t order = 3; order < 6; ++order) {
    INFO("Order:= " << order);
    {
      const size_t number_of_points = 22 - 3 * order;
      single_call(number_of_points, order, 1);
      single_call(number_of_points, order, 2);
    }
    {
      const size_t number_of_points = 22 - 2 * order;
      single_call(number_of_points, order, 3);
    }
    if (order > 3) {
      const size_t number_of_points = 22 - order;
      single_call(number_of_points, order, 4);
      single_call(number_of_points, order, 5);
    }
  }
}
