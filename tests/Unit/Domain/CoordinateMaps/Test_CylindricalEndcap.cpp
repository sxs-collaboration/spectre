// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <random>

#include "Domain/CoordinateMaps/CylindricalEndcap.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

namespace domain {
namespace {
void test_cylindrical_endcap() {
  INFO("CylindricalEndcap");
  // Set up random number generator
  MAKE_GENERATOR(gen);

  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random centers for sphere_one and sphere_two
  const std::array<double, 3> center_one = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE_PRECISE(center_one);
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE_PRECISE(center_two);
  const double dist_between_spheres =
      sqrt(square(center_two[0] - center_one[0]) +
           square(center_two[1] - center_one[1]) +
           square(center_two[2] - center_one[2]));

  // Pick radius of sphere_one not too small compared to the distance
  // between the centers.
  const double radius_one = 0.3 * dist_between_spheres + unit_dis(gen);
  CAPTURE_PRECISE(radius_one);

  // Make sure z_plane intersects sphere_one on the +z side of the
  // center. We don't allow the plane to be displaced by less than 10%
  // or more than 90% of the radius.
  const double z_plane =
      center_one[2] + (0.1 + 0.8 * unit_dis(gen)) * radius_one;
  CAPTURE_PRECISE(z_plane);

  // Now construct sphere_two which we make sure encloses sphere_one,
  // but doesn't have too large of a radius.
  const double radius_two =
      (unit_dis(gen) + 1.0) * (radius_one + dist_between_spheres);
  CAPTURE_PRECISE(radius_two);

  const std::array<double, 3> proj_center = [
    &z_plane, &center_one, &radius_one, &center_two, &radius_two, &gen,
    &unit_dis, &angle_dis
  ]() noexcept {
    // Consider a cone formed by taking the intersection of z_plane and
    // sphere_one (this is a circle called circle_one) and connecting it to
    // center_one. Compute theta, where 2*theta is the opening angle of this
    // cone. Call this cone 'cone_one'.
    const double cos_theta = (z_plane - center_one[2]) / radius_one;

    // We will construct two new cones. The first we will call
    // cone_regular. It is constructed so that cone_regular and
    // cone_one intersect each other on circle_one at right angles.
    // Cone_regular opens in the -z direction with opening angle 2*(pi/2-theta).
    // The apex of cone_regular is cone_regular_apex, defined as follows:
    const std::array<double, 3> cone_regular_apex = {
        center_one[0], center_one[1], center_one[2] + radius_one / cos_theta};
    // A necessary condition for the map being
    // invertible is that proj_center must lie either inside
    // cone_regular (which opens in the -z direction and intersects
    // cone_one), or proj_center must lie inside the reflection of
    // cone_regular (which opens in the +z direction).  If proj_center
    // is inside cone_regular (i.e. the one that opens in the -z
    // direction), an additional condition for the map being
    // invertible is that proj_center cannot lie between sphere_one
    // and the center of cone_regular.

    // The second cone we will construct we will call cone_opening.
    // It will have some maximum opening angle that we choose freely.
    const double max_opening_angle = M_PI / 3.0;
    // max_opening_angle determines the maximum possible x-coordinate
    // of proj_center.
    const double max_proj_center_z =
        z_plane -
        radius_one * sqrt(1.0 - square(cos_theta)) / tan(max_opening_angle);
    // Cone_opening has apex
    // (center_one[0], center_one[1], max_proj_center_z). Note that if
    // max_opening_angle > pi/4 and if z_plane > center_one[2] then
    // the apex of cone_opening is inside sphere_one.

    // Now that we have two cones, cone_regular and cone_opening.  We
    // will choose proj_center so that it lies inside of both cones,
    // and that it lies inside of sphere_two.
    const double cone_regular_opening_angle = asin(cos_theta);

    if (max_opening_angle < cone_regular_opening_angle) {
      // The easier case.  We need to worry only about max_opening_angle.

      // Choose some smaller cone inside of cone_opening.
      // We will place proj_center on this smaller cone.
      const double beta = unit_dis(gen) * max_opening_angle;

      // Choose an azimuthal coordinate for the point on the cone.
      const double phi = angle_dis(gen);

      // Choose a radius for the point on the cone that will be
      // proj_center. The point should be inside sphere_two.  So
      // determine the radius at which the point intersects
      // sphere_two. This radius is the solution of a quadratic equation
      // ax^2 + bx + c = 0
      const double a = 1.0;
      const double b =
          2.0 * (-(max_proj_center_z - center_two[2]) * cos(beta) +
                 (center_one[0] - center_two[0]) * sin(beta) * cos(phi) +
                 (center_one[1] - center_two[1]) * sin(beta) * sin(phi));
      const double c = square(max_proj_center_z - center_two[2]) +
                       square(center_one[0] - center_two[0]) +
                       square(center_one[1] - center_two[1]) -
                       square(radius_two);
      ASSERT(c < 0.0,
             "max_proj_center_z is too negative. The apex "
             "is not inside sphere_two");
      // Should be two real roots. Choose the positive one.
      const double r_max = positive_root(a, b, c);

      const double proj_radius = unit_dis(gen) * r_max;
      return std::array<double, 3>{
          center_one[0] + proj_radius * sin(beta) * cos(phi),
          center_one[1] + proj_radius * sin(beta) * sin(phi),
          max_proj_center_z - proj_radius * cos(beta)};
    } else {
      // The more difficult case.

      // We may need to try several values of alpha.
      // When one works, exit.
      while (true) {
        // Choose some smaller cone inside of cone_regular.
        // We will place proj_center on this smaller cone.
        // We do not allow the smaller cone to go all the way to
        // cone_regular_opening_angle.
        const double alpha = unit_dis(gen) * 0.95 * cone_regular_opening_angle;

        // Consider the circle at which the smaller cone intersects
        // cone_opening, and find the distance from cone_regular_apex to
        // this circle.
        const double radius_coord_circle =
            (cone_regular_apex[2] - max_proj_center_z) *
            tan(max_opening_angle) / (tan(max_opening_angle) - tan(alpha)) /
            cos(alpha);

        // Choose an azimuthal coordinate for the point on the cone.
        const double phi = angle_dis(gen);

        // Choose a radius for the point on the cone that will be
        // proj_center. The point should be inside sphere_two.  So
        // determine the radius at which the point intersects
        // sphere_two. This radius is the solution of a quadratic equation
        // ax^2 + bx + c = 0
        const double a = 1.0;
        const double b =
            2.0 * (-(cone_regular_apex[2] - center_two[2]) * cos(alpha) +
                   (center_one[0] - center_two[0]) * sin(alpha) * cos(phi) +
                   (center_one[1] - center_two[1]) * sin(alpha) * sin(phi));
        const double c = square(cone_regular_apex[2] - center_two[2]) +
                         square(center_one[0] - center_two[0]) +
                         square(center_one[1] - center_two[1]) -
                         square(radius_two);

        double x0 = std::numeric_limits<double>::signaling_NaN();
        double x1 = std::numeric_limits<double>::signaling_NaN();
        const int num_real_roots = gsl_poly_solve_quadratic(a, b, c, &x0, &x1);
        double r_max = 0.0;
        if (num_real_roots == 2) {
          // Take the largest root.  The smallest one is negative if
          // cone_regular_apex[0] is inside sphere_two, positive if
          // cone_regular_apex[0] is outside sphere_two.
          r_max = std::max(x0, x1);
        } else if (num_real_roots == 1) {
          r_max = x0;
        } else {
          ASSERT(false, "No roots were found. Something is horribly wrong.");
        }

        if (r_max <= radius_coord_circle) {
          // We cannot satisfy all the conditions because sphere_two is
          // too small. So try again with a different alpha.
          continue;
        }
        const double proj_radius =
            radius_coord_circle + unit_dis(gen) * (r_max - radius_coord_circle);
        return std::array<double, 3>{
            center_one[0] + proj_radius * sin(alpha) * cos(phi),
            center_one[1] + proj_radius * sin(alpha) * sin(phi),
            cone_regular_apex[2] - proj_radius * cos(alpha)};
      }
    }
  }
  ();
  CAPTURE_PRECISE(proj_center);

  const CoordinateMaps::CylindricalEndcap map(
      center_one, center_two, proj_center, radius_one, radius_two, z_plane);
  test_suite_for_map_on_cylinder(map);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.CylindricalEndcap",
                  "[Domain][Unit]") {
  test_cylindrical_endcap();
  CHECK(not CoordinateMaps::CylindricalEndcap{}.is_identity());
}
}  // namespace domain
