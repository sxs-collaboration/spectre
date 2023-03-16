// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Shape.SphereTransition",
                  "[Domain][Unit]") {
  SphereTransition sphere_transition{2., 4.};
  const std::array<double, 3> lower_bound{{2., 0., 0.}};
  CHECK(sphere_transition(lower_bound) == approx(1.));
  const std::array<double, 3> midpoint{{3., 0., 0.}};
  CHECK(sphere_transition(midpoint) == approx(1. / 3.));
  const std::array<double, 3> upper_bound{{4., 0., 0.}};
  CHECK(sphere_transition(upper_bound) == approx(0.));
  const std::array<double, 3> below_lower_bound{{1., 0., 0.}};
  CHECK(sphere_transition(below_lower_bound) == approx(1.));
  const std::array<double, 3> above_upper_bound{{5., 0., 0.}};
  CHECK(sphere_transition(above_upper_bound) == approx(0.));
}

}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
