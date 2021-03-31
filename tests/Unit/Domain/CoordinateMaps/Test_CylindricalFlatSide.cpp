// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <random>

#include "Domain/CoordinateMaps/CylindricalFlatSide.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

namespace domain {
namespace {
void test_cylindrical_flat_side() {
  INFO("CylindricalFlatSide");
  // Set up random number generator
  MAKE_GENERATOR(gen);

  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose a random center and radius for sphere_two
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);
  const double radius_two = 0.3 + unit_dis(gen);
  CAPTURE(radius_two);

  // Choose a random center for the annulus, making sure the
  // z-coordinate is outside sphere_two and at least 5 percent below
  // it.  Also make sure that the center of the annulus is not offset
  // in x-y farther than the radius of sphere_two.
  const std::array<double, 3> center_one = [&radius_two, &center_two, &unit_dis,
                                            &angle_dis, &gen]() noexcept {
    const double rho = unit_dis(gen) * radius_two;
    const double phi = angle_dis(gen);
    const double x = center_two[0] + rho * cos(phi);
    const double y = center_two[1] + rho * sin(phi);
    const double z = center_two[2] - radius_two * (1.05 + unit_dis(gen));
    return std::array<double, 3>{{x, y, z}};
  }();
  CAPTURE(center_one);

  // Pick proj_center inside sphere_two.
  // Don't let proj_center be too close to the edge of the sphere.
  const std::array<double, 3> proj_center = [&center_two, &radius_two, &gen,
                                             &unit_dis, &interval_dis,
                                             &angle_dis]() noexcept {
    const double phi = angle_dis(gen);
    const double cos_theta = interval_dis(gen);
    const double sin_theta = sqrt(1.0 - square(cos_theta));
    const double r = radius_two * 0.95 * unit_dis(gen);
    return std::array<double, 3>{center_two[0] + r * sin_theta * cos(phi),
                                 center_two[1] + r * sin_theta * sin(phi),
                                 center_two[2] + r * cos_theta};
  }();
  CAPTURE(proj_center);

  const double dist_annulus_proj = sqrt(square(center_one[0] - proj_center[0]) +
                                        square(center_one[1] - proj_center[1]) +
                                        square(center_one[2] - proj_center[2]));

  // Pick outer radius of annulus, but not too small.
  // Smallness is decided by making sure angle subtended by annulus with
  // respect to projection center is greater than 0.05 radians.
  const double outer_radius = dist_annulus_proj * 0.05 + unit_dis(gen);
  CAPTURE(outer_radius);

  // Pick inner radius of annulus.
  // Don't make the annulus too thin,
  // and don't make the inner radius too small.
  const double min_inner_radius_fac =
      std::max(0.05, dist_annulus_proj * 0.01 / outer_radius);
  const double max_inner_radius_fac = 0.95 - min_inner_radius_fac;
  const double inner_radius =
      outer_radius *
      (min_inner_radius_fac + max_inner_radius_fac * unit_dis(gen));
  CAPTURE(inner_radius);

  const CoordinateMaps::CylindricalFlatSide map(center_one, center_two,
                                                proj_center, inner_radius,
                                                outer_radius, radius_two);
  test_suite_for_map_on_cylinder(map, 1.0, 2.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.CylindricalFlatSide",
                  "[Domain][Unit]") {
  test_cylindrical_flat_side();
  CHECK(not CoordinateMaps::CylindricalFlatSide{}.is_identity());
}
}  // namespace domain
