// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <random>

#include "Domain/CoordinateMaps/CylindricalFlatEndcap.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

namespace domain {
namespace {
void test_cylindrical_flat_endcap() {
  INFO("CylindricalFlatEndcap");
  // Set up random number generator
  MAKE_GENERATOR(gen);

  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose random radii for sphere_two and circle_one, and ensure
  // that the two radii don't have a ratio of more than 10:1.
  const double radius_two = 0.1 + 0.9 * unit_dis(gen);
  CAPTURE(radius_two);
  const double radius_one = 0.1 + 0.9 * unit_dis(gen);
  CAPTURE(radius_one);

  // Choose a random center for sphere_two.
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);

  // Choose a random center for the circle, making sure the z-coordinate
  // is outside sphere_two and below it by at least 5% of radius_two
  // and at most 5 times radius_two.
  // For the x and y coordinates, make sure that they are not displaced by
  // more than radius_one + radius_two with respect to sphere_two.
  const std::array<double, 3> center_one = {
      center_two[0] + (radius_one + radius_two) * interval_dis(gen),
      center_two[1] + (radius_one + radius_two) * interval_dis(gen),
      center_two[2] - radius_two -
          5.0 * radius_two * (0.01 + 0.99 * unit_dis(gen))};
  CAPTURE(center_one);

  // Pick proj_center inside sphere_two, but less than 95% from the
  // surface.
  const std::array<double, 3> proj_center = [&center_two, &radius_two, &gen,
                                             &unit_dis, &interval_dis,
                                             &angle_dis]() {
    const double phi = angle_dis(gen);
    const double cos_theta = interval_dis(gen);
    const double sin_theta = sqrt(1.0 - square(cos_theta));
    const double r = radius_two * 0.95 * unit_dis(gen);
    return std::array<double, 3>{center_two[0] + r * sin_theta * cos(phi),
                                 center_two[1] + r * sin_theta * sin(phi),
                                 center_two[2] + r * cos_theta};
  }();
  CAPTURE(proj_center);

  const CoordinateMaps::CylindricalFlatEndcap map(
      center_one, center_two, proj_center, radius_one, radius_two);
  test_suite_for_map_on_cylinder(map, 0.0, 1.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.CylindricalFlatEndcap",
                  "[Domain][Unit]") {
  test_cylindrical_flat_endcap();
  CHECK(not CoordinateMaps::CylindricalFlatEndcap{}.is_identity());
}
}  // namespace domain
