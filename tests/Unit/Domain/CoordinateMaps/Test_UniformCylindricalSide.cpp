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
        {{center_two[0], center_two[1], -center_two[2]}},
        radius_one,
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.UniformCylindricalSide",
                  "[Domain][Unit]") {
  test_uniform_cylindrical_side();
  test_uniform_cylindrical_side_planes_equal(false);
  test_uniform_cylindrical_side_planes_equal(true);
  CHECK(not CoordinateMaps::UniformCylindricalSide{}.is_identity());
}
}  // namespace domain
