// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Shape.SphereTransition",
                  "[Domain][Unit]") {
  SphereTransition sphere_transition{2., 4.};
  constexpr double eps = std::numeric_limits<double>::epsilon() * 100;
  const std::array<double, 3> lower_bound{{2., 0., 0.}};
  CHECK(sphere_transition(lower_bound) == approx(1.));
  const std::array<double, 3> lower_bound_eps{{2. - eps, 0., 0.}};
  CHECK(sphere_transition(lower_bound_eps) == approx(1.));
  const std::array<double, 3> midpoint{{3., 0., 0.}};
  CHECK(sphere_transition(midpoint) == approx(1. / 3.));
  const std::array<double, 3> upper_bound{{4., 0., 0.}};
  CHECK(sphere_transition(upper_bound) == approx(0.));
  const std::array<double, 3> upper_bound_eps{{4. + eps, 0., 0.}};
  CHECK(sphere_transition(upper_bound_eps) == approx(0.));
}

}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
