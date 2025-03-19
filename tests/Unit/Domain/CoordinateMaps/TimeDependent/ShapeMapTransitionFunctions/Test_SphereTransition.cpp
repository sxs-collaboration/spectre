// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Shape.SphereTransition",
                  "[Domain][Unit]") {
  constexpr double eps = std::numeric_limits<double>::epsilon() * 100;
  const size_t l_max = 4;
  const size_t num_coefs = 2 * square(l_max + 1);
  const double time = 1.3;

  domain::FunctionsOfTimeMap functions_of_time{};
  functions_of_time["Shape"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          0.0, std::array{DataVector{num_coefs, 1.e-3}}, 10.0);

  const std::vector<std::array<double, 3>> test_points{
      {2., 0., 0.},
      {(1.0 - eps) * 2., 0., 0.},
      {3., 0., 0.},
      {4., 0., 0.},
      {(1.0 + eps) * 4., 0., 0.}};

  {
    INFO("Sphere transition");
    SphereTransition sphere_transition{2., 4.};
    sphere_transition = serialize_and_deserialize(sphere_transition);
    SphereTransition sphere_transition_interior{2., 4., false, true};
    sphere_transition_interior =
        serialize_and_deserialize(sphere_transition_interior);
    const domain::CoordinateMaps::TimeDependent::Shape shape_map{
        std::array{0.0, 0.0, 0.0}, l_max, l_max,
        std::make_unique<SphereTransition>(sphere_transition), "Shape"};
    const domain::CoordinateMaps::TimeDependent::Shape shape_map_interior{
        std::array{0.0, 0.0, 0.0}, l_max, l_max,
        std::make_unique<SphereTransition>(sphere_transition_interior),
        "Shape"};

    const std::vector<double> function_values{0.5, 0.5, 0.5 / 3.0, 0.0, 0.0};

    for (size_t i = 0; i < test_points.size(); i++) {
      CAPTURE(test_points[i]);
      CAPTURE(function_values[i]);
      CHECK(sphere_transition(test_points[i], std::nullopt) ==
            approx(function_values[i]));
      const double radius = std::max(magnitude(test_points[i]), 2.0);
      CHECK(sphere_transition(test_points[i], {1}) ==
            approx(function_values[i] / radius));
      test_inverse_map(shape_map, test_points[i], time, functions_of_time);
    }

    const std::vector<std::array<double, 3>> interior_points{{1.0, 0.0, 0.0},
                                                             {0.0, 0.0, 0.0}};
    const std::vector<double> interior_function_values{0.125, 0.0};
    for (size_t i = 0; i < interior_points.size(); i++) {
      CAPTURE(interior_points[i]);
      CHECK(sphere_transition_interior(interior_points[i], std::nullopt) ==
            interior_function_values[i]);
      test_inverse_map(shape_map_interior, interior_points[i], time,
                       functions_of_time);
    }

    // Check close, but not at, center and r_min
    test_inverse_map(shape_map_interior, std::array{2.0 * eps, 0.0, 0.0}, time,
                     functions_of_time);
    test_inverse_map(shape_map_interior,
                     std::array{(1.0 - eps) * 2.0, 0.0, 0.0}, time,
                     functions_of_time);
  }
  {
    INFO("Reverse sphere transition");
    SphereTransition sphere_transition{2., 4., true};
    sphere_transition = serialize_and_deserialize(sphere_transition);

    const domain::CoordinateMaps::TimeDependent::Shape shape_map{
        std::array{0.0, 0.0, 0.0}, 4, 4,
        std::make_unique<SphereTransition>(sphere_transition), "Shape"};

    const std::vector<double> function_values{0.0, 0.0, 0.5 / 3.0, 0.25, 0.25};

    for (size_t i = 0; i < test_points.size(); i++) {
      CAPTURE(test_points[i]);
      CAPTURE(function_values[i]);
      CHECK(sphere_transition(test_points[i], std::nullopt) ==
            approx(function_values[i]));
      const double radius = std::min(magnitude(test_points[i]), 4.0);
      CHECK(sphere_transition(test_points[i], {1}) ==
            approx(function_values[i] /
                   (equal_within_roundoff(radius, 0.0) ? 1.0 : radius)));
      test_inverse_map(shape_map, test_points[i], time, functions_of_time);
    }
  }
}

}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
