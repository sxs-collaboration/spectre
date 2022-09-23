// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "Domain/CoordinateMaps/UniformCylindricalEndcap.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"

namespace domain {

namespace {

double maximum_rho_for_invertibility(double rho_max_so_far, double center_one_z,
                                     const std::array<double, 3>& center_two,
                                     double radius_one, double radius_two,
                                     double theta_max_one,
                                     double theta_max_two) {
  const auto is_invertible = [&](const double rho) {
    // Direction shouldn't matter here, so just choose x-axis
    return CoordinateMaps::
                   is_uniform_cylindrical_endcap_invertible_on_sphere_one(
                       {{center_two[0] + rho, center_two[1], center_one_z}},
                       center_two, radius_one, radius_two, theta_max_one,
                       theta_max_two)
               ? 1.0
               : -1.0;
  };
  const double invertible_at_zero = is_invertible(0.0);
  const double invertible_at_max = is_invertible(rho_max_so_far);
  ASSERT(invertible_at_zero == 1.0, "Map should always be invertible at rho=0");
  if (invertible_at_zero * invertible_at_max > 0.0) {
    // Root is not bracketed, so map is invertible everywhere.
    return rho_max_so_far;
  }

  try {
    rho_max_so_far =
        // NOLINTNEXTLINE(clang-analyzer-core)
        RootFinder::toms748(is_invertible, 0.0, rho_max_so_far,
                            invertible_at_zero, invertible_at_max, 1.e-4,
                            1.e-4);
    rho_max_so_far = std::max(0.0, rho_max_so_far - 1.e-4);
  } catch (std::exception&) {
    CHECK(false);
  }
  return rho_max_so_far;
}

void test_uniform_cylindrical_endcap() {
  // Here sphere_two is contained in sphere_one.
  INFO("UniformCylindricalEndcap");

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
  constexpr double min_angle = 0.075;
  constexpr double max_angle_increment = 0.45 - min_angle;

  // Make sure z_plane_two intersects sphere_two on the +z side of the
  // center. We don't allow the plane to be too close to the center or
  // too close to the edge.
  const double z_plane_two =
      center_two[2] +
      cos((min_angle + max_angle_increment * unit_dis(gen)) * M_PI) *
          radius_two;
  CAPTURE(z_plane_two);

  // Choose z_plane_frac_one = (z_plane_one - center_one[2])/radius_one
  const double z_plane_frac_one =
      cos((min_angle + max_angle_increment * unit_dis(gen)) * M_PI);

  // Compute the minimum allowed value of the angle alpha.
  const double theta_max_one = acos(z_plane_frac_one);
  const double theta_max_two = acos((z_plane_two - center_two[2]) / radius_two);
  const double min_alpha = 1.1 * std::max(theta_max_one, theta_max_two);

  // Choose a random radius of sphere_one, not too small and not larger
  // than sphere_two.
  const double radius_one = [&center_two, &radius_two, &z_plane_frac_one,
                             &z_plane_two, &min_alpha, &theta_max_one,
                             &theta_max_two, &unit_dis, &gen]() {
    // max_radius_one_to_fit_inside_sphere_two is the largest that
    // radius_one can be and still satisfy both
    // 0.98 radius_two >= radius_one + |C_1-C_2| and
    // z_plane_two >= z_plane_one + 0.04*r_2.
    // If radius_one takes on that value, then center_one-center_two
    // must point in the minus-z direction, and only one value of
    // center_one[2] is allowed.
    const double max_radius_one_to_fit_inside_sphere_two =
        (0.94 * radius_two + z_plane_two - center_two[2]) /
        (1.0 + z_plane_frac_one);

    // max_radius_one_for_alpha is the largest that radius_one can be
    // and still satisfy alpha > min_alpha.  For tan(min_alpha) > 0,
    // if max_radius_one_to_fit_inside_sphere_two is satisfied, then
    // alpha > min_alpha imposes no additional restriction on radius.
    const double max_radius_one_for_alpha =
        min_alpha > 0.5 * M_PI
            ? std::numeric_limits<double>::max()
            : std::min(radius_two * sin(theta_max_two) / sin(theta_max_one),
                       (0.98 * radius_two - z_plane_two + center_two[2] -
                        radius_two * sin(theta_max_two) * tan(min_alpha)) /
                           (1.0 - cos(theta_max_one) -
                            sin(theta_max_one) * tan(min_alpha)));
    CHECK(max_radius_one_for_alpha > 0.0);
    // We add an additional safety factor of 0.99 to
    // max_radius_one_to_fit_inside_sphere_two so that radius_one
    // doesn't get quite that large.
    const double max_radius_one = std::min(
        {0.98 * radius_two, 0.99 * max_radius_one_to_fit_inside_sphere_two,
         max_radius_one_for_alpha});
    const double min_radius_one = 0.05 * radius_two;
    CHECK(max_radius_one >= min_radius_one);
    return min_radius_one + unit_dis(gen) * (max_radius_one - min_radius_one);
  }();
  CAPTURE(radius_one);

  // Choose a random z-center of sphere_one.
  const double center_one_z =
      [&radius_two, &radius_one, &z_plane_frac_one, &center_two, &z_plane_two,
       &min_alpha, &theta_max_one, &theta_max_two, &unit_dis, &gen]() {
        const double max_center_one_z_from_alpha =
            (tan(min_alpha) <= 0.0 or
             radius_one * sin(theta_max_one) <= radius_two * sin(theta_max_two))
                ? std::numeric_limits<double>::max()
                : (radius_two * sin(theta_max_two) -
                   radius_one * sin(theta_max_one)) *
                      tan(min_alpha);
        // max_center_one_z comes from the restriction
        // z_plane_two >= z_plane_one + 0.04*radius_two, and the restriction
        // 0.98 r_2 >= r_1 + | C_1 - C_2 |
        const double max_center_one_z = std::min(
            {max_center_one_z_from_alpha,
             z_plane_two - z_plane_frac_one * radius_one - 0.04 * radius_two,
             center_two[2] + 0.98 * radius_two - radius_one});
        // min_center_one_z comes from the restriction 0.98 r_2 >= r_1 +
        // |C_1-C_2|
        const double min_center_one_z =
            center_two[2] - 0.98 * radius_two + radius_one;
        CHECK(min_center_one_z <= max_center_one_z);
        return min_center_one_z +
               unit_dis(gen) * (max_center_one_z - min_center_one_z);
      }();

  // Now we can compute z_plane_one
  const double z_plane_one = center_one_z + radius_one * z_plane_frac_one;
  CAPTURE(z_plane_one);

  // Only thing remaining are the x and y centers of sphere_one.
  const double horizontal_distance_spheres = [&z_plane_one, &z_plane_two,
                                              &theta_max_one, &theta_max_two,
                                              &min_alpha, &center_one_z,
                                              &center_two, &radius_one,
                                              &radius_two, &unit_dis, &gen]() {
    // Let rho be the horizontal (x-y) distance between the centers of
    // the spheres.

    // maximum rho so that sphere2 is inside of sphere 1, with a
    // safety factor of 0.98
    const double max_rho_sphere = sqrt(square(0.98 * radius_two - radius_one) -
                                       square(center_one_z - center_two[2]));

    // Alpha always gets smaller when rho gets larger (for other
    // quantities fixed). So if alpha < min_alpha even when rho=0, then
    // there is no hope.  We always fail.
    const double alpha_if_rho_is_zero =
        atan2(z_plane_two - z_plane_one, radius_one * sin(theta_max_one) -
                                             radius_two * sin(theta_max_two));
    CHECK(alpha_if_rho_is_zero >= min_alpha);

    const double max_rho_alpha_first_term =
        abs(min_alpha - 0.5 * M_PI) < 1.e-4
            ? 0.0
            : (z_plane_two - z_plane_one) / tan(min_alpha);

    // maximum rho so that the alpha condition is satisfied
    const double max_rho_alpha = max_rho_alpha_first_term -
                                 radius_one * sin(theta_max_one) +
                                 radius_two * sin(theta_max_two);
    const double max_rho = maximum_rho_for_invertibility(
        std::min(max_rho_sphere, max_rho_alpha), center_one_z, center_two,
        radius_one, radius_two, theta_max_one, theta_max_two);
    return unit_dis(gen) * max_rho;
  }();

  const double phi = angle_dis(gen);
  const std::array<double, 3> center_one = {
      center_two[0] + horizontal_distance_spheres * cos(phi),
      center_two[1] + horizontal_distance_spheres * sin(phi), center_one_z};
  CAPTURE(center_one);

  CHECK(CoordinateMaps::is_uniform_cylindrical_endcap_invertible_on_sphere_one(
      center_one, center_two, radius_one, radius_two,
      acos((z_plane_one - center_one[2]) / radius_one),
      acos((z_plane_two - center_two[2]) / radius_two)));

  const CoordinateMaps::UniformCylindricalEndcap map(
      center_one, center_two, radius_one, radius_two, z_plane_one, z_plane_two);
  test_suite_for_map_on_cylinder(map, 0.0, 1.0, true);

  // The following are tests that the inverse function correctly
  // returns an invalid std::optional when called for a point that is
  // outside the range of the map.

  // Point with z less than z_plane_one.
  CHECK_FALSE(map.inverse({{0.0,0.0,z_plane_one-1.0}}));

  // Point outside sphere_two
  CHECK_FALSE(map.inverse(
      {{center_two[0], center_two[1], center_two[2] + 1.2 * radius_two}}));

  // Point inside sphere_one (but z>z_plane_one since z_plane_one
  // intersects sphere_one)
  CHECK_FALSE(map.inverse(
      {{center_one[0], center_one[1], center_one[2] + 0.99 * radius_one}}));

  // Point outside the cone
  CHECK_FALSE(map.inverse(
      {{center_one[0], center_one[1] + radius_one * sin(theta_max_one) * 1.01,
        z_plane_one + (z_plane_one - center_one[2]) * 1.e-5}}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.UniformCylindricalEndcap",
                  "[Domain][Unit]") {
  test_uniform_cylindrical_endcap();
  CHECK(not CoordinateMaps::UniformCylindricalEndcap{}.is_identity());
}
}  // namespace domain
