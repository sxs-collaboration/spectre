// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"

namespace domain {

namespace {

double maximum_rho_for_invertibility(double rho_max_so_far, double z_center_two,
                                     const std::array<double, 3>& center_one,
                                     double radius_one, double radius_two,
                                     double theta_max_one) {
  // is_invertible returns a double and not a bool because it will be
  // used in a root-finder below.
  const auto is_invertible = [&](const double rho) {
    // Direction shouldn't matter here, so just choose x-axis
    return CoordinateMaps::
                   is_uniform_cylindrical_flat_endcap_invertible_on_sphere_one(
                       center_one,
                       {{center_one[0] + rho, center_one[1], z_center_two}},
                       radius_one, radius_two, theta_max_one)
               ? 1.0
               : -1.0;
  };
  const double invertible_at_zero = is_invertible(0.0);
  const double invertible_at_max = is_invertible(rho_max_so_far);
  // Sanity check here... Should always be invertible at zero.
  CHECK(invertible_at_zero == 1.0);
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

void test_uniform_cylindrical_flat_endcap() {
  // Here sphere_two is contained in sphere_one.
  INFO("UniformCylindricalFlatEndcap");

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random center for sphere_one
  const std::array<double, 3> center_one = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_one);

  // Choose a random radius of sphere_one.
  const double radius_one = 2.0 * (unit_dis(gen) + 1.0);
  CAPTURE(radius_one);

  // These angles describe how close the z-planes can be to the
  // centers or edges of the spheres.
  constexpr double min_angle = 0.075;
  constexpr double max_angle_increment = 0.35 - min_angle;

  // Make sure z_plane_two intersects sphere_two on the +z side of the
  // center. We don't allow the plane to be too close to the center or
  // too close to the edge.
  const double z_plane_one =
      center_one[2] +
      cos((min_angle + max_angle_increment * unit_dis(gen)) * M_PI) *
          radius_one;
  CAPTURE(z_plane_one);

  // Compute z_center_two.
  // z_center_two > C_1^z + 1.05*radius_one.
  // z_center_two < C_1^z + 5*radius_one.
  const double z_center_two =
      center_one[2] + radius_one * (unit_dis(gen) * 3.95 + 1.05);

  // Compute the minimum allowed value of the angle alpha.
  const double theta_max_one = acos((z_plane_one - center_one[2]) / radius_one);
  const double min_alpha = 1.1 * theta_max_one;

  // Compute the radius of disk_two
  const double radius_two = [&z_plane_one, &z_center_two, &radius_one,
                             &min_alpha, &theta_max_one, &unit_dis, &gen]() {
    const double max_radius_two = 2.0 * sin(theta_max_one) * radius_one;
    const double min_radius_two_if_rho_is_zero =
        min_alpha > 0.5 * M_PI
            ? std::numeric_limits<double>::min()
            : sin(theta_max_one) * radius_one -
                  (z_center_two - z_plane_one) / tan(min_alpha);
    const double min_radius_two = std::max(
        min_radius_two_if_rho_is_zero, 0.1 * sin(theta_max_one) * radius_one);
    CHECK(max_radius_two >= min_radius_two);
    return min_radius_two + unit_dis(gen) * (max_radius_two - min_radius_two);
  }();
  CAPTURE(radius_two);

  // Compute rho = horizontal distance between the centers.
  const double horizontal_distance = [&radius_one, &radius_two, &z_plane_one,
                                      &z_center_two, &center_one, &min_alpha,
                                      &theta_max_one, &unit_dis, &gen]() {
    const double max_rho_alpha =
        min_alpha > 0.5 * M_PI
            ? std::numeric_limits<double>::max()
            : (z_center_two - z_plane_one) / tan(min_alpha) -
                  sin(theta_max_one) * radius_one + radius_two;
    const double max_rho_offset = sin(theta_max_one) * radius_one;
    const double max_rho = maximum_rho_for_invertibility(
        std::min(max_rho_offset, max_rho_alpha), z_center_two, center_one,
        radius_one, radius_two, theta_max_one);
    return unit_dis(gen) * max_rho;
  }();

  const double phi = angle_dis(gen);
  const std::array<double, 3> center_two = {
      center_one[0] + horizontal_distance * cos(phi),
      center_one[1] + horizontal_distance * sin(phi), z_center_two};
  CAPTURE(center_two);

  const CoordinateMaps::UniformCylindricalFlatEndcap map(
      center_one, center_two, radius_one, radius_two, z_plane_one);
  test_suite_for_map_on_cylinder(map, 0.0, 1.0, true);

  // The following are tests that the inverse function correctly
  // returns an invalid std::optional when called for a point that is
  // outside the range of the map.

  // Point with z less than z_plane_one.
  CHECK_FALSE(map.inverse({{0.0, 0.0, z_plane_one - 1.0}}).has_value());

  // Point above disk.
  CHECK_FALSE(map.inverse({{center_two[0], center_two[1],
                            center_two[2] + 1.2 * radius_two}})
                  .has_value());

  // Point inside sphere_one (but z>z_plane_one since z_plane_one
  // intersects sphere_one)
  CHECK_FALSE(map.inverse({{center_one[0], center_one[1],
                            center_one[2] + 0.99 * radius_one}})
                  .has_value());

  // Point outside the cone
  CHECK_FALSE(
      map.inverse({{center_one[0],
                    center_one[1] + radius_one * sin(theta_max_one) * 1.01,
                    z_plane_one + (z_plane_one - center_one[2]) * 1.e-5}})
          .has_value());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.UniformCylindricalFlatEndcap",
                  "[Domain][Unit]") {
  test_uniform_cylindrical_flat_endcap();
  CHECK(not CoordinateMaps::UniformCylindricalFlatEndcap{}.is_identity());
}
}  // namespace domain
