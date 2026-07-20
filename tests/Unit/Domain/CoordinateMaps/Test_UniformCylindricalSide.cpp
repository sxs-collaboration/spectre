// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"

namespace domain {

namespace {

void test_uniform_cylindrical_side_planes_equal(
    const bool flip_z_axis = false) {
  INFO("UniformCylindricalSidePlanesEqual");

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random center for sphere_two
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);

  // Choose a random radius of sphere_two, reasonably large.
  const double radius_two = 6.0 * (unit_dis(gen) + 1.0);
  CAPTURE(radius_two);

  // Choose z_plane_frac_plus_one=(z_plane_plus_one-center_one[2])/radius_one
  // Note here that max_angle_one_plus is > 0.5, so that
  // z_plane_plus_one can be at a lower value of z than center_one[2],
  // and thus z_plane_frac_plus_one may be positive or negative.
  const double min_angle_one_plus = 0.15;
  const double max_angle_one_plus = 0.59;
  const double angle_one_plus =
      min_angle_one_plus +
      (max_angle_one_plus - min_angle_one_plus) * unit_dis(gen);
  const double z_plane_frac_plus_one = cos(angle_one_plus * M_PI);

  // Choose z_plane_frac_minus_one=(z_plane_minus_one-center_one[2])/radius_one
  // (note that this quantity is < 0).
  const double min_angle_one_minus = 0.15;
  const double max_angle_one_minus = angle_one_plus > 0.4 ? 0.3 : 0.4;
  // Note that we deliberately choose max_angle_one_plus +
  // max_angle_one_minus < 1.  This ensures that z_plane_frac_plus_one
  // > z_plane_frac_minus_one, which is important for
  // max_radius_one_planes below.
  const double z_plane_frac_minus_one =
      -cos((min_angle_one_minus +
            (max_angle_one_minus - min_angle_one_minus) * unit_dis(gen)) *
           M_PI);

  // Choose an angle for the positive z-plane
  // Don't go too close to the edge if angle_one_plus is large.
  const double min_angle_shared = angle_one_plus > 0.4 ? 0.25 : 0.15;
  const double max_angle_shared = 0.75;
  const double z_plane_plus_two =
      center_two[2] +
      cos((min_angle_shared +
           (max_angle_shared - min_angle_shared) * unit_dis(gen)) *
          M_PI) *
          radius_two;
  const double z_plane_plus_one = z_plane_plus_two;
  CAPTURE(z_plane_plus_two);
  CAPTURE(z_plane_plus_one);

  // Choose an angle for the negative z-plane for sphere 2
  // Note that min_angle_two must be < 1-max_angle_shared
  // (otherwise we cannot fit a sphere_one)
  // max_angle_two comes from the requirement that
  // z_plane_minus_two < z_plane_plus_two - 0.18 radius_two.
  const double min_angle_two = 0.15;
  const double max_angle_two =
      center_two[2] < z_plane_plus_two
          ? 0.4
          : std::min(0.4, acos((center_two[2] - z_plane_plus_two) / radius_two +
                               0.18) /
                              M_PI);
  CHECK(min_angle_two < max_angle_two);

  const double z_plane_minus_two =
      center_two[2] -
      cos((min_angle_two + (max_angle_two - min_angle_two) * unit_dis(gen)) *
          M_PI) *
          radius_two;
  CAPTURE(z_plane_minus_two);

  // Choose radius of sphere_one.
  const double radius_one = [&z_plane_frac_plus_one, &z_plane_frac_minus_one,
                             &radius_two, &center_two, &z_plane_plus_one,
                             &z_plane_minus_two, &unit_dis, &gen]() {
    // max_radius_one_planes is determined by the condition
    // z_plane_minus_one >= z_plane_minus_two + 0.03 * radius_two
    // The expression below is derived by
    // evaluating z_plane_minus_one using the formula for
    // z_plane_frac_minus_one, and eliminating center_one[2] using the formula
    // for z_plane_frac_plus_one.
    const double max_radius_one_planes =
        (z_plane_plus_one - z_plane_minus_two - 0.03 * radius_two) /
        (z_plane_frac_plus_one - z_plane_frac_minus_one);

    // max_radius_one_fit is determined by the condition that
    // 0.98 radius_two >= radius_one + |C_1-C_2|,
    // eliminating center_one[2] using the formula
    // for z_plane_frac_plus_one.
    const double max_radius_one_fit =
        std::min((0.98 * radius_two - center_two[2] + z_plane_plus_one) /
                     (1.0 + z_plane_frac_plus_one),
                 (0.98 * radius_two + center_two[2] - z_plane_plus_one) /
                     (1.0 - z_plane_frac_plus_one));

    // Compute the minimum allowed value of the angle alpha_minus.
    // (these quantities are measured from zero; note the minus signs)
    const double theta_max_minus_one = acos(-z_plane_frac_minus_one);
    const double theta_max_minus_two =
        acos(-(z_plane_minus_two - center_two[2]) / radius_two);
    const double min_alpha_minus = 1.1 * theta_max_minus_one;
    // max_radius_one_from_alpha comes from the restriction
    // that alpha > min_alpha_minus
    // and the expression for z_plane_frac_plus_one (used to eliminate
    // center_one[2]) and the expression for z_plane_frac_minus_one.
    const double max_radius_one_from_alpha =
        (z_plane_plus_one - z_plane_minus_two +
         radius_two * tan(min_alpha_minus) * sin(theta_max_minus_two)) /
        (tan(min_alpha_minus) * sin(theta_max_minus_one) +
         z_plane_frac_plus_one - z_plane_frac_minus_one);
    const double max_radius_one = std::min(
        {max_radius_one_planes, max_radius_one_fit, max_radius_one_from_alpha});
    const double min_radius_one = 0.08 * radius_two;
    CHECK(max_radius_one >= min_radius_one);
    return min_radius_one + unit_dis(gen) * (max_radius_one - min_radius_one);
  }();
  CAPTURE(radius_one);

  const std::array<double, 3> center_one = {
      center_two[0], center_two[1],
      z_plane_plus_two - z_plane_frac_plus_one * radius_one};
  CAPTURE(center_one);

  const double z_plane_minus_one =
      center_one[2] + radius_one * z_plane_frac_minus_one;
  CAPTURE(z_plane_minus_one);

  if (flip_z_axis) {
    // Here we test the map with z_plane_minus_one equal to
    // z_plane_minus_two.  We do this by simply flipping the map
    // parameters about the z axis (and exchanging parameters named
    // plus and minus).  This way we don't need to rewrite the entire
    // test for z_plane_minus_one==z_plane_minus_two.
    const CoordinateMaps::UniformCylindricalSide map(
        {{center_one[0], center_one[1], -center_one[2]}},
        {{center_two[0], center_two[1], -center_two[2]}}, radius_one,
        radius_two, -z_plane_minus_one, -z_plane_plus_one, -z_plane_minus_two,
        -z_plane_plus_two);
    test_suite_for_map_on_cylinder(map, 1.0, 2.0, true, true);
  } else {
    const CoordinateMaps::UniformCylindricalSide map(
        center_one, center_two, radius_one, radius_two, z_plane_plus_one,
        z_plane_minus_one, z_plane_plus_two, z_plane_minus_two);
    test_suite_for_map_on_cylinder(map, 1.0, 2.0, true, true);
  }
}

void test_uniform_cylindrical_side() {
  INFO("UniformCylindricalSide");

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random center for sphere_two
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);

  // Choose a random radius of sphere_two, reasonably large.
  const double radius_two = 6.0 * (unit_dis(gen) + 1.0);
  CAPTURE(radius_two);

  // These angles describe how close the z-planes can be to the
  // centers or edges of the spheres.
  const double min_angle = 0.15;
  const double max_angle = 0.4;

  // Make sure z_plane_plus_two intersects sphere_two on the +z side of the
  // center. We don't allow the plane to be too close to the center or
  // too close to the edge.
  const double z_plane_plus_two =
      center_two[2] +
      cos((min_angle + (max_angle - min_angle) * unit_dis(gen)) * M_PI) *
          radius_two;
  CAPTURE(z_plane_plus_two);

  // Make sure z_plane_minus_two intersects sphere_two on the -z side of the
  // center. We don't allow the plane to be too close to the center or
  // too close to the edge.
  const double z_plane_minus_two =
      center_two[2] -
      cos((min_angle + (max_angle - min_angle) * unit_dis(gen)) * M_PI) *
          radius_two;
  CAPTURE(z_plane_minus_two);

  // Choose z_plane_frac_plus_one=(z_plane_plus_one-center_one[2])/radius_one
  const double z_plane_frac_plus_one =
      cos((min_angle + (max_angle - min_angle) * unit_dis(gen)) * M_PI);

  // Choose
  // z_plane_frac_minus_one=(z_plane_minus_one-center_one[2])/radius_one (note
  // that this quantity is < 0).
  const double z_plane_frac_minus_one =
      -cos((min_angle + (max_angle - min_angle) * unit_dis(gen)) * M_PI);

  // Compute the minimum allowed value of the angle alpha_plus.
  const double theta_max_plus_one = acos(z_plane_frac_plus_one);
  const double theta_max_plus_two =
      acos((z_plane_plus_two - center_two[2]) / radius_two);
  const double min_alpha_plus =
      1.1 * std::max(theta_max_plus_one, theta_max_plus_two);

  // Compute the minimum allowed value of the angle alpha_minus.
  // (these quantities are measured from zero; note the minus signs)
  const double theta_max_minus_one = acos(-z_plane_frac_minus_one);
  const double theta_max_minus_two =
      acos(-(z_plane_minus_two - center_two[2]) / radius_two);
  const double min_alpha_minus =
      1.1 * std::max(theta_max_minus_one, theta_max_minus_two);

  // Choose a random radius of sphere_one, not too small and not larger
  // than sphere_two.
  const double radius_one = [&center_two, &radius_two, &z_plane_frac_plus_one,
                             &z_plane_plus_two, &min_alpha_plus,
                             &theta_max_plus_one, &theta_max_plus_two,
                             &z_plane_frac_minus_one, &z_plane_minus_two,
                             &min_alpha_minus, &theta_max_minus_one,
                             &theta_max_minus_two, &unit_dis, &gen]() {
    const double z_upper_separation = 0.03;
    const double z_lower_separation = 0.03;
    // max_radius_one_to_fit_inside_sphere_two_plus is the largest that
    // radius_one can be and still satisfy both
    // 0.98 radius_two >= radius_one + |C_1-C_2| and
    // z_plane_plus_two >= z_plane_plus_one + z_upper_separation*radius_two
    // when center_one_z is unknown and z_plane_plus_one is unknown
    // (but the quantity z_plane_frac_plus_one is known).
    // This value comes about when center_one and center_two have the
    // same x and y components, and when center_one_z < center_two_z, and
    // when center_one_z takes on its largest possible value consistent with
    // 0.98 radius_two >= radius_one + |C_1-C_2|.
    // The latter condition is C^z_1 >= radius_one + C^z_2 - 0.98 radius_two.
    //
    // Similarly, max_radius_one_to_fit_inside_sphere_two_minus
    // is the largest that radius_one can be and still satisfy both
    // 0.98 radius_two >= radius_one + |C_1-C_2| and
    // z_plane_minus_two <= z_plane_minus_one - z_lower_separation*radius_two
    // when center_one_z is unknown and z_plane_minus_one is unknown
    // (but the quantity z_plane_frac_minus_one is known).
    // This value comes about when center_one and center_two have the
    // same x and y components, and when center_one_z > center_two_z, and
    // when center_one_z takes on its smallest possible value consistent with
    // 0.98 radius_two >= radius_one + |C_1-C_2|.
    // The latter condition is C^z_1 <= -radius_one + C^z_2 + 0.98 radius_two.
    //
    // Here we take the min of both of the above quantities.
    const double max_radius_one_to_fit_inside_sphere_two =
        std::min((z_plane_plus_two - center_two[2] +
                  (0.98 - z_upper_separation) * radius_two) /
                     (z_plane_frac_plus_one + 1.0),
                 (z_plane_minus_two - center_two[2] -
                  (0.98 - z_lower_separation) * radius_two) /
                     (z_plane_frac_minus_one - 1.0));
    // max_radius_one_for_alpha_minus is the largest that radius_one can be
    // and still satisfy alpha_minus > min_alpha_minus.  For
    // tan(min_alpha_minus) > 0, if max_radius_one_to_fit_inside_sphere_two is
    // satisfied, then alpha_minus > min_alpha_minus imposes no additional
    // restriction on radius.
    const double max_radius_one_for_alpha_plus =
        min_alpha_plus > 0.5 * M_PI
            ? std::numeric_limits<double>::max()
            : std::min(
                  radius_two * sin(theta_max_plus_two) /
                      sin(theta_max_plus_one),
                  (0.98 * radius_two - z_plane_plus_two + center_two[2] -
                   radius_two * sin(theta_max_plus_two) * tan(min_alpha_plus)) /
                      (1.0 - cos(theta_max_plus_one) -
                       sin(theta_max_plus_one) * tan(min_alpha_plus)));
    const double max_radius_one_for_alpha_minus =
        min_alpha_minus > 0.5 * M_PI
            ? std::numeric_limits<double>::max()
            : std::min(radius_two * sin(theta_max_minus_two) /
                           sin(theta_max_minus_one),
                       (0.98 * radius_two + z_plane_minus_two - center_two[2] -
                        radius_two * sin(theta_max_minus_two) *
                            tan(min_alpha_minus)) /
                           (1.0 - cos(theta_max_minus_one) -
                            sin(theta_max_minus_one) * tan(min_alpha_minus)));
    CHECK(max_radius_one_for_alpha_minus > 0.0);
    CHECK(max_radius_one_for_alpha_plus > 0.0);
    // max_radius_one_to_fit_between_plane_twos is the maximum radius_one
    // that satisfies the two conditions
    // z_plane_plus_two >= z_plane_plus_one + z_upper_separation*radius_two
    // z_plane_minus_two <= z_plane_minus_one - z_lower_separation*radius_two
    //
    // This condition is derived from noting that
    // z_plane_plus_one = center_one[2]+radius_one*z_plane_frac_plus_one
    // and z_plane_minus_one = center_one[2]+radius_one*z_plane_frac_minus_one
    // (recall z_plane_frac_minus_one is negative)
    // and noting that the max value of center_one[2] is >= the min value
    // of center_one[2].
    const double max_radius_one_to_fit_between_plane_twos =
        (z_plane_plus_two - z_plane_minus_two -
         (z_upper_separation + z_lower_separation) * radius_two) /
        (z_plane_frac_plus_one - z_plane_frac_minus_one);
    CHECK(max_radius_one_to_fit_between_plane_twos > 0.0);

    double max_radius_one = std::min(
        {0.98 * radius_two, max_radius_one_to_fit_inside_sphere_two,
         max_radius_one_to_fit_between_plane_twos,
         max_radius_one_for_alpha_minus, max_radius_one_for_alpha_plus});
    double min_radius_one = 0.08 * radius_two;

    CHECK(max_radius_one >= min_radius_one);
    return min_radius_one + unit_dis(gen) * (max_radius_one - min_radius_one);
  }();
  CAPTURE(radius_one);

  // Choose a random z-center of sphere_one.
  const double center_one_z = [&radius_two, &radius_one, &center_two,
                               &z_plane_frac_plus_one, &z_plane_plus_two,
                               &min_alpha_plus, &theta_max_plus_one,
                               &theta_max_plus_two, &z_plane_frac_minus_one,
                               &z_plane_minus_two, &min_alpha_minus,
                               &theta_max_minus_one, &theta_max_minus_two,
                               &unit_dis, &gen]() {
    const double max_center_one_z_from_alpha_plus =
        (tan(min_alpha_plus) <= 0.0 or radius_one * sin(theta_max_plus_one) <=
                                           radius_two * sin(theta_max_plus_two))
            ? std::numeric_limits<double>::max()
            : (radius_two * sin(theta_max_plus_two) -
               radius_one * sin(theta_max_plus_one)) *
                  tan(min_alpha_plus);
    // Note minus sign in min_center_one_z_from_alpha_minus
    const double min_center_one_z_from_alpha_minus =
        (tan(min_alpha_minus) <= 0.0 or
         radius_one * sin(theta_max_minus_one) <=
             radius_two * sin(theta_max_minus_two))
            ? std::numeric_limits<double>::lowest()
            : -(radius_two * sin(theta_max_minus_two) -
                radius_one * sin(theta_max_minus_one)) *
                  tan(min_alpha_minus);
    CHECK(min_center_one_z_from_alpha_minus <=
          max_center_one_z_from_alpha_plus);
    // max_center_one_z comes from the restriction
    // z_plane_plus_two >= z_plane_plus_one + 0.03*radius_two,
    // and the restriction
    // 0.98 r_2 >= r_1 + | C_1 - C_2 |
    // and the restriction
    // C^z_1 < C^z_2 + r_1 + r_2/5
    // which is designed to not allow a tiny sphere 1 at the edge of
    // a large sphere 2.
    const double max_center_one_z =
        std::min({max_center_one_z_from_alpha_plus,
                  z_plane_plus_two - z_plane_frac_plus_one * radius_one -
                      0.03 * radius_two,
                  center_two[2] + radius_one + 0.2 * radius_two,
                  center_two[2] + 0.98 * radius_two - radius_one});
    // min_center_one_z comes from the restriction
    // z_plane_minus_two <= z_plane_minus_one - 0.03*radius_two,
    // and the restriction
    // 0.98 r_2 >= r_1 + |C_1 - C_2 |
    // and the restriction
    // C^z_1 > C^z_2 - r_1 - r_2/5
    // which is designed to not allow a tiny sphere 1 at the edge of
    // a large sphere 2.
    const double min_center_one_z =
        std::max({min_center_one_z_from_alpha_minus,
                  z_plane_minus_two - z_plane_frac_minus_one * radius_one +
                      0.03 * radius_two,
                  center_two[2] - radius_one - 0.2 * radius_two,
                  center_two[2] - 0.98 * radius_two + radius_one});
    CHECK(min_center_one_z <= max_center_one_z);
    return min_center_one_z +
           unit_dis(gen) * (max_center_one_z - min_center_one_z);
  }();
  CAPTURE(center_one_z);

  // Now we can compute z_plane_plus_one and z_plane_minus_one
  const double z_plane_plus_one =
      center_one_z + radius_one * z_plane_frac_plus_one;
  CAPTURE(z_plane_plus_one);
  const double z_plane_minus_one =
      center_one_z + radius_one * z_plane_frac_minus_one;
  CAPTURE(z_plane_minus_one);

  // Only thing remaining are the x and y centers of sphere_one.
  const double horizontal_distance_spheres =
      [&z_plane_plus_one, &z_plane_plus_two, &theta_max_plus_one,
       &theta_max_plus_two, &min_alpha_plus, &z_plane_minus_one,
       &z_plane_minus_two, &theta_max_minus_one, &theta_max_minus_two,
       &min_alpha_minus, &center_one_z, &center_two, &radius_one, &radius_two,
       &unit_dis, &gen]() {
        // Let rho be the horizontal (x-y) distance between the centers of
        // the spheres.

        // maximum rho obeying the condition
        // 0.98 R_2 <= R_1 + |C_1-C_2|
        const double max_rho_sphere =
            sqrt(square(0.98 * radius_two - radius_one) -
                 square(center_one_z - center_two[2]));

        // We don't want a tiny sphere 1 all the way on the edge of sphere 2.
        // So demand that at least some of sphere_one lies along the polar
        // axis of sphere_two.
        const double max_rho_sphere2 = radius_one;

        // We demand that the edge of sphere 1 is not too close to the
        // edge of sphere 2.  But don't let max_rho_sphere3 be negative.
        const double max_rho_sphere3 =
            std::max(0.0, radius_two * 0.95 - radius_one);

        // Alpha always gets smaller when rho gets larger (for other
        // quantities fixed). So if alpha < min_alpha even when rho=0, then
        // there is no hope.  We always fail.
        const double alpha_plus_if_rho_is_zero =
            atan2(z_plane_plus_two - z_plane_plus_one,
                  radius_one * sin(theta_max_plus_one) -
                      radius_two * sin(theta_max_plus_two));
        CHECK(alpha_plus_if_rho_is_zero >= min_alpha_plus);
        const double alpha_minus_if_rho_is_zero =
            atan2(z_plane_minus_one - z_plane_minus_two,
                  radius_one * sin(theta_max_minus_one) -
                      radius_two * sin(theta_max_minus_two));
        CHECK(alpha_minus_if_rho_is_zero >= min_alpha_minus);

        const double max_rho_alpha_plus_first_term =
            abs(min_alpha_plus - 0.5 * M_PI) < 1.e-4
                ? 0.0
                : (z_plane_plus_two - z_plane_plus_one) / tan(min_alpha_plus);

        const double max_rho_alpha_minus_first_term =
            abs(min_alpha_minus - 0.5 * M_PI) < 1.e-4
                ? 0.0
                : (z_plane_minus_one - z_plane_minus_two) /
                      tan(min_alpha_minus);

        // maximum rho so that the alpha condition is satisfied
        const double max_rho_alpha_plus = max_rho_alpha_plus_first_term -
                                          radius_one * sin(theta_max_plus_one) +
                                          radius_two * sin(theta_max_plus_two);
        const double max_rho_alpha_minus =
            max_rho_alpha_minus_first_term -
            radius_one * sin(theta_max_minus_one) +
            radius_two * sin(theta_max_minus_two);
        const double max_rho =
            std::min({max_rho_sphere, max_rho_sphere2, max_rho_sphere3,
                      max_rho_alpha_plus, max_rho_alpha_minus});
        CHECK(max_rho >= 0.0);
        return unit_dis(gen) * max_rho;
      }();

  const double phi = angle_dis(gen);
  const std::array<double, 3> center_one = {
      center_two[0] + horizontal_distance_spheres * cos(phi),
      center_two[1] + horizontal_distance_spheres * sin(phi), center_one_z};
  CAPTURE(center_one);

  const CoordinateMaps::UniformCylindricalSide map(
      center_one, center_two, radius_one, radius_two, z_plane_plus_one,
      z_plane_minus_one, z_plane_plus_two, z_plane_minus_two);
  test_suite_for_map_on_cylinder(map, 1.0, 2.0, true, true);

  // The following are tests that the inverse function correctly
  // returns an invalid std::optional when called for a point that is
  // outside the range of the map.

  // Point with z > z_plane_plus_two.
  CHECK_FALSE(map.inverse({{0.0, 0.0, z_plane_plus_two + 1.0}}).has_value());

  // Point with z < z_plane_minus_two.
  CHECK_FALSE(map.inverse({{0.0, 0.0, z_plane_minus_two - 1.0}}).has_value());

  // Point outside sphere_two
  CHECK_FALSE(map.inverse({{center_two[0], center_two[1] + 1.01 * radius_two,
                            center_two[2]}})
                  .has_value());

  // Point inside sphere_one (but z_plane_minus_one<z<z_plane_plus_one
  // intersects sphere_one)
  CHECK_FALSE(map.inverse({{center_one[0], center_one[1],
                            0.5 * (z_plane_plus_one + z_plane_minus_one)}})
                  .has_value());

  // Point inside the northern cone
  if (z_plane_plus_two != z_plane_plus_one) {
    CHECK_FALSE(map.inverse({{center_two[0],
                              center_two[1] +
                                  radius_two * sin(theta_max_plus_two) * 0.98,
                              z_plane_plus_two -
                                  (z_plane_plus_two - center_two[2]) * 1.e-5}})
                    .has_value());
  }

  // Point inside the southern cone
  if (z_plane_minus_two != z_plane_minus_one) {
    CHECK_FALSE(map.inverse({{center_two[0],
                              center_two[1] +
                                  radius_two * sin(theta_max_minus_two) * 0.98,
                              z_plane_minus_two +
                                  (center_two[2] - z_plane_minus_two) * 1.e-5}})
                    .has_value());
  }
}
void test_uniform_cylindrical_side_class_b() {
  INFO("UniformCylindricalSideClassB");
  // Class B: z_minus_one < z_minus_two < z_plus_one < z_plus_two.
  // The outer sphere's lower cut sits above the inner sphere's lower cut,
  // so the mapped z-bands of the two spheres overlap rather than nesting.
  // Class A (standard, z_minus_two < z_minus_one) is tested by
  // test_uniform_cylindrical_side().

  // Concrete case 1: concentric spheres.
  //   center_one = center_two = {0,0,0}, R1=0.9, R2=3.
  //   z_minus_one = R1*cos(theta_max), z_minus_two = -0.4 (above z_minus_one).
  {
    INFO("Concentric spheres");
    const double r1 = 0.9;
    const double r2 = 3.0;
    const double theta_min = 0.2 * M_PI;
    const double theta_max = 0.75 * M_PI;
    const double z_plus_one = r1 * cos(theta_min);
    const double z_minus_one = r1 * cos(theta_max);
    const double z_plus_two = r2 * cos(theta_min);
    const double z_minus_two = -0.4;
    const CoordinateMaps::UniformCylindricalSide map(
        {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, r1, r2, z_plus_one, z_minus_one,
        z_plus_two, z_minus_two);
    test_suite_for_map_on_cylinder(map, 1.0, 2.0, true, true);

    // z below z_minus_one is out of range in Class B
    CHECK_FALSE(map.inverse({{0.0, 0.0, z_minus_one - 1.0}}).has_value());
    // z above z_plus_two is out of range
    CHECK_FALSE(map.inverse({{0.0, 0.0, z_plus_two + 1.0}}).has_value());
    // outside sphere_two
    CHECK_FALSE(map.inverse({{0.0, r2 + 0.1, 0.0}}).has_value());
    // inside sphere_one
    CHECK_FALSE(map.inverse({{0.0, 0.0, 0.0}}).has_value());
    // inside northern cone: z in [z_plus_one, z_plus_two], small rho
    CHECK_FALSE(map.inverse({{0.0, 0.1, z_plus_one + 0.01}}).has_value());
    // Point inside sphere_one: z just above z_minus_one with small rho gives
    // r < r1 (caught by the sphere_one radius check, not the cone check).
    CHECK_FALSE(map.inverse({{0.0, 0.2, z_minus_one + 0.01}}).has_value());
    // Point outside the Class B upward cone at z in [z_minus_one, z_minus_two].
    // This is the specific fix from point_outside_cone: a point with rho much
    // larger than circle_radius needs lambda > lambda_max (=> zbar < -1) and
    // is not in the image.  Before the fix this would ERROR with "Root is not
    // bracketed"; now it correctly returns nullopt.
    // At z=-0.5: circle_radius ≈ 1.98 (interpolating between r1*sin(90°)=0.9
    // and r2*sin(theta_max_two)≈2.97), so rho=2.5 is well outside.
    const double z_cone_test = -0.5;
    const double rho_cone_test = 2.5;
    CHECK_FALSE(map.inverse({{0.0, rho_cone_test, z_cone_test}}).has_value());
    // Valid interior point at z in [z_minus_one, z_minus_two] with small rho
    // (inside the Class B cone).  This must NOT return nullopt — before the
    // fix, the wrong point_inside_cone check would have incorrectly rejected
    // it. (rhobar=1.1, phi=pi/2, zbar=-0.9 maps into the cone interior.)
    const double rhobar_valid = 1.1;
    const double zbar_valid = -0.9;
    const auto valid_class_b_pt =
        map(std::array<double, 3>{{0.0, rhobar_valid, zbar_valid}});
    CHECK(map.inverse(valid_class_b_pt).has_value());
  }

  // Concrete case 2: offset z-centers.
  //   center_one = {0,0,center_one_z}, R1=1.1; center_two = {0,0,0}, R2=4.
  //   z_minus_two = center_one_z (above z_minus_one).
  {
    INFO("Offset z-centers");
    const double r1 = 1.1;
    const double r2 = 4.0;
    const double center_one_z = 0.5;
    const double theta_upper = 0.25 * M_PI;
    const double theta_lower = 0.75 * M_PI;
    const double z_plus_one = center_one_z + r1 * cos(theta_upper);
    const double z_minus_one = center_one_z + r1 * cos(theta_lower);
    const double z_plus_two = r2 * cos(theta_upper);
    const double z_minus_two = center_one_z;
    const CoordinateMaps::UniformCylindricalSide map(
        {{0.0, 0.0, center_one_z}}, {{0.0, 0.0, 0.0}}, r1, r2, z_plus_one,
        z_minus_one, z_plus_two, z_minus_two);
    test_suite_for_map_on_cylinder(map, 1.0, 2.0, true, true);
  }

  // Concrete case 3: Class C — completely separated z-bands
  // (BLeftHollowCylinder-like from the Pill domain).
  // z_plus_one < z_minus_two: the inner sphere's full band lies entirely below
  // the outer sphere's lower z-cut, so there is no z-overlap between the bands.
  // Parameters match Pill with CenterA=0.5, CenterB=-0.70,
  // WedgeInnerRadius=0.6, WedgeOuterRadius=1.1, CylinderOuterRadius=3.  Inner
  // sphere B after the minus-x rotation: center=(0,0,0.7), R_B=1.1. Inner band
  // [z_minus_one, z_plus_one] = [0.7, 1.3]. Outer band [z_minus_two,
  // z_plus_two] ≈ [1.3786, 2.4470].
  {
    INFO("Separated z-bands (BLeftHollowCylinder-like, Class C)");
    const double r1 = 1.1;
    const double r2 = 3.0;
    const double center_one_z = 0.7;
    const double z_plus_one = 1.3;
    const double z_minus_one = center_one_z;
    const double z_minus_two = 1.3786;
    const double z_plus_two = 2.4470;
    const CoordinateMaps::UniformCylindricalSide bleft_map(
        {{0.0, 0.0, center_one_z}}, {{0.0, 0.0, 0.0}}, r1, r2, z_plus_one,
        z_minus_one, z_plus_two, z_minus_two);
    test_suite_for_map_on_cylinder(bleft_map, 1.0, 2.0, true, true);

    // Below z_minus_one.
    CHECK_FALSE(bleft_map.inverse({{0.0, 0.0, z_minus_one - 0.2}}).has_value());
    // Above z_plus_two.
    CHECK_FALSE(bleft_map.inverse({{0.0, 0.0, z_plus_two + 0.1}}).has_value());
    // Inside sphere_one: z at sphere center, r < r1.
    CHECK_FALSE(
        bleft_map.inverse({{0.0, 0.0, center_one_z + 0.3}}).has_value());
    // Outside sphere_two.
    CHECK_FALSE(bleft_map.inverse({{0.0, r2 + 0.1, 0.0}}).has_value());
    // Inside the upper cone: z > z_plus_one, small rho.
    CHECK_FALSE(bleft_map.inverse({{0.0, 0.1, z_plus_one + 0.05}}).has_value());

    // Point outside the Class B upward cone at z in [z_minus_one, z_minus_two].
    // This exercises the point_outside_cone fix.
    // Before the fix, the inverse would ERROR ("Root is not bracketed");
    // now it correctly returns nullopt.
    // At z=0.8: lambda_tilde=(0.8-0.7)/(1.3786-0.7)=0.147;
    //   circle_radius≈1.1*(0.853)+2.664*(0.147)≈0.938+0.392=1.330.
    // rho=2.5 >> 1.330 → outside cone → nullopt.
    const double z_cone_test = 0.8;
    const double rho_large = 2.5;
    CHECK_FALSE(bleft_map.inverse({{0.0, rho_large, z_cone_test}}).has_value());
    // Also near z_minus_one: the image there is a thin annulus near rho≈r1.
    CHECK_FALSE(
        bleft_map.inverse({{0.0, 2.0, z_minus_one + 0.01}}).has_value());

    // Valid interior point with z in [z_minus_one, z_minus_two] and small rho
    // (inside the Class B cone).  Must NOT return nullopt.
    // (rhobar=1.05, phi=pi/2, zbar=-0.9 maps to z≈0.765 ∈ [0.7, 1.3786],
    //  rho≈1.18 < circle_radius≈1.25 → inside cone.)
    const double rhobar_valid = 1.05;
    const double zbar_valid = -0.9;
    const auto valid_bleft_pt =
        bleft_map(std::array<double, 3>{{0.0, rhobar_valid, zbar_valid}});
    CHECK(bleft_map.inverse(valid_bleft_pt).has_value());
  }

  // Randomized test: aligned x,y centers.
  // With horizontal_dist=0:
  //   alpha_minus = pi/2 > 1.1*(pi-theta_max_one) for theta_max_one in
  //     [0.62pi, 0.83pi] (since pi - theta_max_one < 0.38pi < pi/2).
  //   alpha_plus = atan2(positive, negative) > pi/2 > 1.1*theta_min_one for
  //     theta_min_one in [0.2pi, 0.38pi].
  // Both alpha conditions are automatically satisfied, greatly simplifying
  // parameter generation.
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);

  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);
  const double radius_two = 6.0 * (unit_dis(gen) + 1.0);
  CAPTURE(radius_two);
  // radius_one in [0.09, 0.39] * radius_two
  const double radius_one_min_frac = 0.09;
  const double radius_one_range_frac = 0.3;
  const double radius_one =
      (radius_one_min_frac + radius_one_range_frac * unit_dis(gen)) *
      radius_two;
  CAPTURE(radius_one);

  // center_one has same x,y as center_two; z-offset in [-z_offset,+z_offset].
  const double z_offset_frac = 0.15;
  const double center_one_z =
      center_two[2] + (2.0 * unit_dis(gen) - 1.0) * z_offset_frac * radius_two;
  const std::array<double, 3> center_one = {center_two[0], center_two[1],
                                            center_one_z};
  CAPTURE(center_one);

  // Upper cut of sphere_two: theta_min_two in [theta_min_two_lo,
  // theta_min_two_hi].
  const double theta_min_two_lo = 0.2;
  const double theta_min_two_hi = 0.38;
  const double z_plus_two =
      center_two[2] +
      radius_two * cos((theta_min_two_lo +
                        (theta_min_two_hi - theta_min_two_lo) * unit_dis(gen)) *
                       M_PI);
  CAPTURE(z_plus_two);

  // Lower cut of sphere_one: theta_max_one in [theta_max_one_lo,
  // theta_max_one_hi].
  const double theta_max_one_lo = 0.62;
  const double theta_max_one_hi = 0.83;
  const double z_minus_one =
      center_one_z +
      radius_one * cos((theta_max_one_lo +
                        (theta_max_one_hi - theta_max_one_lo) * unit_dis(gen)) *
                       M_PI);
  CAPTURE(z_minus_one);

  // Upper cut of sphere_one: must satisfy z_plus_one <= z_plus_two -
  // z_sep*R2 and theta_min_one in [theta_min_one_lo, theta_min_one_hi].
  const double z_sep_frac = 0.04;
  const double theta_min_one_lo = 0.2;
  const double theta_min_one_hi = 0.38;
  const double cos_theta_min_one_upper = std::min(
      cos(theta_min_one_lo * M_PI),
      (z_plus_two - z_sep_frac * radius_two - center_one_z) / radius_one);
  const double cos_theta_min_one_lower = cos(theta_min_one_hi * M_PI);
  if (cos_theta_min_one_upper < cos_theta_min_one_lower) {
    // Parameter draw infeasible; skip this run.
    return;
  }
  const double z_plus_one =
      center_one_z + radius_one * (cos_theta_min_one_lower +
                                   unit_dis(gen) * (cos_theta_min_one_upper -
                                                    cos_theta_min_one_lower));
  CAPTURE(z_plus_one);

  // Class B lower cut of sphere_two: z_minus_two in
  //   [z_minus_one + z_sep*R2, min(z_plus_one - z_sep*R2,
  //                                center_two[2] +
  //                                cos(theta_max_two_upper*pi)*R2)].
  // The upper bound keeps cos_theta_max_two < cos(theta_max_two_upper*pi).
  const double theta_max_two_upper = 0.15;
  const double z_minus_two_lo = z_minus_one + z_sep_frac * radius_two;
  const double z_minus_two_hi =
      std::min(z_plus_one - z_sep_frac * radius_two,
               center_two[2] + cos(theta_max_two_upper * M_PI) * radius_two);
  CAPTURE(z_minus_two_lo);
  CAPTURE(z_minus_two_hi);
  if (z_minus_two_lo >= z_minus_two_hi) {
    return;  // Degenerate draw; skip.
  }
  const double z_minus_two =
      z_minus_two_lo + unit_dis(gen) * (z_minus_two_hi - z_minus_two_lo);
  CAPTURE(z_minus_two);

  const CoordinateMaps::UniformCylindricalSide map(
      center_one, center_two, radius_one, radius_two, z_plus_one, z_minus_one,
      z_plus_two, z_minus_two);
  test_suite_for_map_on_cylinder(map, 1.0, 2.0, true, true);

  // Out-of-range inverse checks for Class B.
  // z below z_minus_one (not z_minus_two) is now out of range.
  CHECK_FALSE(map.inverse({{center_two[0], center_two[1], z_minus_one - 1.0}})
                  .has_value());
  CHECK_FALSE(map.inverse({{center_two[0], center_two[1], z_plus_two + 1.0}})
                  .has_value());
  CHECK_FALSE(map.inverse({{center_two[0], center_two[1] + 1.01 * radius_two,
                            center_two[2]}})
                  .has_value());
  CHECK_FALSE(map.inverse({{center_one[0], center_one[1],
                            0.5 * (z_plus_one + z_minus_one)}})
                  .has_value());

  // Northern cone: z just above z_plus_one and near the polar axis is inside
  // the upper cone and outside the mapped region.
  if (z_plus_two != z_plus_one) {
    const double cone_frac = 1.e-4;
    const double z_nc = z_plus_one + cone_frac * (z_plus_two - z_plus_one);
    const double small_rho = 0.01 * radius_two;
    CHECK_FALSE(map.inverse({{center_two[0], center_two[1] + small_rho, z_nc}})
                    .has_value());
  }

  // Class B lower cone: at z = z_minus_one the cone radius equals
  // r1*sin(theta_max_one) < radius_one, so rho = rho_outside_factor*radius_one
  // is always strictly outside the cone at that z.  The point is also always
  // inside sphere_two (verified analytically for the parameter ranges above).
  if (z_minus_two > z_minus_one) {
    const double rho_outside_factor = 1.5;
    CHECK_FALSE(map.inverse({{center_two[0],
                              center_two[1] + rho_outside_factor * radius_one,
                              z_minus_one}})
                    .has_value());
  }

  // Valid point in [z_minus_one, z_minus_two]: the forward map at a point with
  // rhobar in (1, 2) and zbar near the lower edge of logical domain must
  // round-trip correctly through the inverse.
  {
    const double rhobar_valid = 1.1;
    const double zbar_valid = -0.9;
    const auto fwd =
        map(std::array<double, 3>{{0.0, rhobar_valid, zbar_valid}});
    CHECK(map.inverse(fwd).has_value());
  }
}

#ifdef SPECTRE_DEBUG
void test_assert_checks() {
  INFO("UniformCylindricalSideAssertChecks");
  // Concentric Class A baseline: theta_min = 0.2pi, theta_max = 0.8pi for
  // both spheres.  All assertions are satisfied by this configuration; each
  // sub-case below perturbs exactly one parameter to trigger one assertion.
  const std::array<double, 3> c0{0.0, 0.0, 0.0};
  const double r2 = 3.0;
  const double r1 = 0.9;
  const double zp1 = r1 * cos(0.2 * M_PI);
  const double zm1 = r1 * cos(0.8 * M_PI);
  const double zp2 = r2 * cos(0.2 * M_PI);
  const double zm2 = r2 * cos(0.8 * M_PI);

  // Basic radius violations
  CHECK_THROWS_WITH(
      (CoordinateMaps::UniformCylindricalSide(c0, c0, -1.0, r2, zp1, zm1, zp2,
                                              zm2)),
      Catch::Matchers::ContainsSubstring("Cannot have negative radius_one"));
  CHECK_THROWS_WITH(
      (CoordinateMaps::UniformCylindricalSide(c0, c0, r1, -1.0, zp1, zm1, zp2,
                                              zm2)),
      Catch::Matchers::ContainsSubstring("Cannot have negative radius_two"));
  CHECK_THROWS_WITH(
      (CoordinateMaps::UniformCylindricalSide(c0, c0, 0.07 * r2, r2, zp1, zm1,
                                              zp2, zm2)),
      Catch::Matchers::ContainsSubstring("must be >= 0.08 * radius_two"));

  // z-plane separation violations
  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                        c0, c0, r1, r2, zp1, zm1, zp1 + 0.01 * r2, zm2)),
                    Catch::Matchers::ContainsSubstring(
                        "z_plane_plus_two must be >= z_plane_plus_one"));
  CHECK_THROWS_WITH(
      (CoordinateMaps::UniformCylindricalSide(c0, c0, r1, r2, zp1, zm1, zp2,
                                              zm1 - 0.01 * r2)),
      Catch::Matchers::ContainsSubstring(
          "z_plane_minus_two must differ from z_plane_minus_one"));

  // Class B: outer sphere band inverted (z_plus_two < z_minus_two)
  {
    const double zm2_b = 1.0;
    const double zp2_b = 0.9;
    CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                          c0, c0, r1, r2, zp1, zm1, zp2_b, zm2_b)),
                      Catch::Matchers::ContainsSubstring(
                          "must be strictly above z_plane_minus_two"));
  }

  // Sphere-one theta bounds
  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                        c0, c0, r1, r2, r1 * cos(0.45 * M_PI), zm1, zp2, zm2)),
                    Catch::Matchers::ContainsSubstring(
                        "z_plane_plus_one is too close to the center"));
  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                        c0, c0, r1, r2, r1 * cos(0.1 * M_PI), zm1, zp2, zm2)),
                    Catch::Matchers::ContainsSubstring(
                        "z_plane_plus_one is too far from the center"));
  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                        c0, c0, r1, r2, zp1, r1 * cos(0.55 * M_PI), zp2, zm2)),
                    Catch::Matchers::ContainsSubstring(
                        "z_plane_minus_one is too close to the center"));
  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                        c0, c0, r1, r2, zp1, r1 * cos(0.9 * M_PI), zp2, zm2)),
                    Catch::Matchers::ContainsSubstring(
                        "z_plane_minus_one is too far from the center"));

  // Sphere-two theta bounds
  {
    const double r1_s = 0.3;
    const double zp1_s = r1_s * cos(0.2 * M_PI);
    const double zm1_s = r1_s * cos(0.8 * M_PI);
    CHECK_THROWS_WITH(
        (CoordinateMaps::UniformCylindricalSide(c0, c0, r1_s, r2, zp1_s, zm1_s,
                                                r2 * cos(0.45 * M_PI), zm2)),
        Catch::Matchers::ContainsSubstring(
            "z_plane_plus_two is too close to the south pole"));
  }
  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(c0, c0, r1, r2, zp1,
                                                            zm1, zp2, -0.85)),
                    Catch::Matchers::ContainsSubstring(
                        "z_plane_minus_two is too close to the north pole"));
  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                        c0, c0, r1, r2, zp1, zm1, r2 * cos(0.1 * M_PI), zm2)),
                    Catch::Matchers::ContainsSubstring(
                        "z_plane_plus_two is too close to the north pole"));
  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                        c0, c0, r1, r2, zp1, zm1, zp2, r2 * cos(0.9 * M_PI))),
                    Catch::Matchers::ContainsSubstring(
                        "z_plane_minus_two is too close to the south pole"));

  // Both pairs of z-planes equal
  CHECK_THROWS_WITH(
      (CoordinateMaps::UniformCylindricalSide(c0, c0, r1, r2, zp1, zm1, zp1,
                                              zm1)),
      Catch::Matchers::ContainsSubstring(
          "Both pairs of z-planes cannot be simultaneously equal"));

  CHECK_THROWS_WITH((CoordinateMaps::UniformCylindricalSide(
                        {{2.5, 0.0, 0.0}}, c0, r1, r2, zp1, zm1, zp2, zm2)),
                    Catch::Matchers::ContainsSubstring(
                        "sufficiently contained inside sphere_two"));
}
#endif

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.UniformCylindricalSide",
                  "[Domain][Unit]") {
  test_uniform_cylindrical_side();
  test_uniform_cylindrical_side_planes_equal(false);
  test_uniform_cylindrical_side_planes_equal(true);
  test_uniform_cylindrical_side_class_b();
#ifdef SPECTRE_DEBUG
  test_assert_checks();
#endif
  CHECK(not CoordinateMaps::UniformCylindricalSide{}.is_identity());
}
}  // namespace domain
