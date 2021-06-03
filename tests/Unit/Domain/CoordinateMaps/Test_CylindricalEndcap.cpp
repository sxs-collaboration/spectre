// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <optional>
#include <random>

#include "Domain/CoordinateMaps/CylindricalEndcap.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

namespace domain {
namespace {

void test_cylindrical_endcap_sphere_two_small() {
  // Here sphere_two is contained in sphere_one.
  INFO("CylindricalEndcapSphereTwoSmall");

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random centers for sphere_one and sphere_two
  const std::array<double, 3> center_one = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_one);
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);

  const double dist_between_spheres =
      sqrt(square(center_two[0] - center_one[0]) +
           square(center_two[1] - center_one[1]) +
           square(center_two[2] - center_one[2]));

  // First pick the radius of the larger sphere to be large enough
  // that the centers of both spheres are reasonably well inside.
  // Also, if dist_between_spheres is really small, then make radius_one
  // larger.
  // Note that choosing 7.0 means that center_two[2]-center_one[2] is
  // less than radius_one/7, so that center_two[2] will be less than
  // z_plane below.
  const double radius_one =
      7.0 * (unit_dis(gen) + 1.0) * std::max(dist_between_spheres, 0.05);
  CAPTURE(radius_one);

  // Make sure z_plane intersects sphere_one on the +z side of the
  // center. We don't allow the plane to be displaced by less than 15%
  // or more than 95% of the radius.
  const double z_plane =
      center_one[2] + (0.15 + 0.8 * unit_dis(gen)) * radius_one;
  CAPTURE(z_plane);

  // Choose radius_two.
  const double radius_two = [&radius_one, &dist_between_spheres, &unit_dis,
                             &gen]() noexcept {
    // Choose radius_two to be small enough that the space between the
    // two spheres is not too cramped. If dist_between_spheres is too
    // small, then limit it as we did when computing radius_one.
    const double max_radius_two =
        0.85 * (radius_one - std::max(dist_between_spheres, 0.05));

    // Choose radius_two to be not too small compared to radius_one.
    const double min_radius_two = 0.1 * radius_one;

    return min_radius_two + unit_dis(gen) * (max_radius_two - min_radius_two);
  }();
  CAPTURE(radius_two);

  // Now choose projection point to be very near the center of sphere_two.
  // And make sure that projection point is less than z_plane.
  const auto proj_center =
      [&center_two, &radius_two, &z_plane, &angle_dis, &unit_dis,
       &gen]() noexcept {
        const double radius =
            std::min(0.1 * radius_two, 0.999 * (z_plane - center_two[2])) *
            unit_dis(gen);
        const double theta = 0.5 * angle_dis(gen);
        const double phi = angle_dis(gen);
        return std::array<double, 3>{
            center_two[0] + radius * sin(theta) * cos(phi),
            center_two[1] + radius * sin(theta) * sin(phi),
            center_two[2] + radius * cos(theta)};
      }();
  CAPTURE(proj_center);

  const CoordinateMaps::CylindricalEndcap map(
      center_one, center_two, proj_center, radius_one, radius_two, z_plane);
  test_suite_for_map_on_cylinder(map, 0.0, 1.0);
}

void test_cylindrical_endcap_sphere_two_large() {
  // Here sphere_one is contained in sphere_two.
  INFO("CylindricalEndcapSphereTwoLarge");

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random centers for sphere_one and sphere_two
  const std::array<double, 3> center_one = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_one);
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);

  // Computes the maximum possible z-coordinate of the projection
  // point, such that proj_center obeys the max_opening_angle
  // restriction.  Consider a cone extending in the minus-z direction
  // from the apex (center_one[0],center_one[1],max_proj_center_z),
  // with an opening angle of max_opening_angle.  proj_center must lie
  // inside that cone.
  auto compute_max_proj_center_z =
      [](double radius_one, double z_plane, double max_opening_angle,
         const std::array<double, 3>& center_one_local) noexcept -> double {
    const double cos_theta = (z_plane - center_one_local[2]) / radius_one;
    return z_plane -
           radius_one * sqrt(1.0 - square(cos_theta)) / tan(max_opening_angle);
  };

  const double dist_between_spheres =
      sqrt(square(center_two[0] - center_one[0]) +
           square(center_two[1] - center_one[1]) +
           square(center_two[2] - center_one[2]));

  // The choice of the maximum opening angle must agree with the
  // value chosen in the sanity checks in CylindricalEndcap.cpp, or
  // else the test will test configurations that do not correspond to
  // those sanity checks.
  // This opening angle is the maximum angle between the z-axis and
  // the line segment that connects the projection point and any
  // point on the circle where the z-plane intersects sphere_one.
  const double max_opening_angle = M_PI / 3.0;

  // Pick the radius of the smaller sphere to be not too small
  // compared to the distance between the centers.  We choose
  // 0.4*dist_between_spheres so that we don't have a microscopic
  // sphere (without that term the radius could be arbitrarily small),
  // but still allow the radius to be smaller than the distance
  // between the centers.
  const double radius_one = 0.4 * dist_between_spheres + unit_dis(gen);
  CAPTURE(radius_one);

  // Make sure z_plane intersects sphere_one on the +z side of the
  // center. We don't allow the plane to be displaced by less than 15%
  // or more than 95% of the radius.
  const double z_plane =
      center_one[2] + (0.15 + 0.8 * unit_dis(gen)) * radius_one;
  CAPTURE(z_plane);

  // Consider a cone formed by taking the intersection of z_plane and
  // sphere_one (this is a circle called circle_one) and connecting it to
  // center_one. Compute theta, where 2*theta is the opening angle of this
  // cone. Call this cone 'cone_one'.
  // cone_one is the cone centered at C_1 in the "Allowed region for P"
  // figure in the doxygen documentation for ClyndricalEndcap.hpp.
  const double cos_theta = (z_plane - center_one[2]) / radius_one;

  // We will construct two new cones. The first we will call
  // cone_regular. It is constructed so that cone_regular and
  // cone_one intersect each other on circle_one at right angles.
  // Cone_regular opens in the -z direction with opening angle
  // 2*(pi/2-theta).
  // cone_regular is the cone centered at S in the "Allowed region for P"
  // figure in the doxygen documentation for ClyndricalEndcap.hpp.
  // The apex of cone_regular is cone_regular_apex,
  // defined as follows:
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

  // The second cone, cone_opening, is the cone extending in the
  // minus-z direction from the apex
  // (center_one[0],center_one[1],max_proj_center_z).  proj_center
  // must lie inside that cone.  Note that if max_opening_angle >
  // pi/4 and if z_plane > center_one[2] then the apex of
  // cone_opening is inside sphere_one.
  const double max_proj_center_z = compute_max_proj_center_z(
      radius_one, z_plane, max_opening_angle, center_one);

  // Now that we have two cones, cone_regular and cone_opening.  We
  // will choose proj_center so that it lies inside of both cones,
  // and that it lies inside of sphere_two.
  // [Note asin(cos_theta) = pi/2-theta]
  const double cone_regular_opening_angle = asin(cos_theta);

  // Because we have two cones, we do the random choice of proj_center
  // and radius_two by rejection.  That is, we choose a proj_center
  // inside cone_opening, and if this point is outside (or sufficiently near)
  // cone_regular we reject proj_center and try again.
  double radius_two = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> proj_center{z_plane, z_plane, z_plane};
  bool found_proj_center = false;
  while (not found_proj_center) {
    // We typically expect success on the first iteration or the first
    // few iterations if we get unlucky.  Let V_O be the volume of
    // cone_opening that is contained inside sphere_two.  If
    // cone_regular (modulo a small buffer angle, the 0.75 below)
    // contains V_O, then success is certain the first
    // time. Otherwise, the probability of success is roughly V_R/V_O,
    // where V_R is the volume of the portion of cone_regular (modulo
    // a small buffer angle, the 0.75 below) that is contained inside
    // V_O.

    // Pick radius of the larger sphere to enclose the smaller one
    // (and not have too large a radius).
    // The 1.01 is there so that the spheres do not almost touch, which
    // causes problems with roundoff.
    radius_two = (unit_dis(gen) + 1.01) * (radius_one + dist_between_spheres);

    // Choose some smaller cone inside of cone_opening.
    // We will place proj_center on this smaller cone.
    const double beta = unit_dis(gen) * max_opening_angle;

    // Choose an azimuthal coordinate for the point on the cone.
    const double phi = angle_dis(gen);

    // Choose the distance proj_dist between the apex of the cone and
    // proj_center. proj_center must be inside sphere_two.  So
    // determine the distance proj_dist_max at which the point intersects
    // sphere_two. This radius is the solution of a quadratic equation
    // a proj_dist_max^2 + b proj_dist_max + c = 0
    const double a = 1.0;
    const double b =
        2.0 * (-(max_proj_center_z - center_two[2]) * cos(beta) +
               (center_one[0] - center_two[0]) * sin(beta) * cos(phi) +
               (center_one[1] - center_two[1]) * sin(beta) * sin(phi));
    const double c = square(max_proj_center_z - center_two[2]) +
                     square(center_one[0] - center_two[0]) +
                     square(center_one[1] - center_two[1]) - square(radius_two);
    ASSERT(c < 0.0,
           "max_proj_center_z is too negative. The apex "
           "is not inside sphere_two");

    // Should be two real roots, one positive and one
    // negative. Choose the positive one.
    const double proj_dist_max = positive_root(a, b, c);

    // Now get the distance along the cone
    const double proj_dist = unit_dis(gen) * proj_dist_max;

    // Now for the rejection.  We reject the point if it is outside
    // cone_opening, and we also reject the point if it is too close
    // to cone_opening (using a factor 0.75 that is the same as the
    // factor in the ASSERTS in the CylindricalEndcap constructor).
    const double alpha = 0.75 * cone_regular_opening_angle;

    // If beta <= alpha, then there is no max distance, i.e.
    // all distances are ok.
    // If beta < alpha, then the max distance is given by
    // (cone_regular_apex[2]-max_proj_center_z)*sin(alpha)/ sin(beta-alpha)
    if (beta <= alpha or
        proj_dist < (cone_regular_apex[2] - max_proj_center_z) * sin(alpha) /
                        sin(beta - alpha)) {
      // accept.
      proj_center = std::array<double, 3>{
          center_one[0] + proj_dist * sin(beta) * cos(phi),
          center_one[1] + proj_dist * sin(beta) * sin(phi),
          max_proj_center_z - proj_dist * cos(beta)};
      found_proj_center = true;
    }
  }
  CAPTURE(radius_two);
  CAPTURE(proj_center);

  const CoordinateMaps::CylindricalEndcap map(
      center_one, center_two, proj_center, radius_one, radius_two, z_plane);
  test_suite_for_map_on_cylinder(map, 0.0, 1.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.CylindricalEndcap",
                  "[Domain][Unit]") {
  test_cylindrical_endcap_sphere_two_large();
  test_cylindrical_endcap_sphere_two_small();
  CHECK(not CoordinateMaps::CylindricalEndcap{}.is_identity());
}
}  // namespace domain
