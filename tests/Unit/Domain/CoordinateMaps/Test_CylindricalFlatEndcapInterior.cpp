// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <random>

#include "Domain/CoordinateMaps/CylindricalFlatEndcapInterior.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace domain {
namespace {

void test_cylindrical_flat_endcap_interior() {
  INFO("CylindricalFlatEndcapInterior");
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // radius_two in [0.5, 1.0] to keep geometry well-conditioned.
  const double radius_two = 0.5 + 0.5 * unit_dis(gen);
  CAPTURE(radius_two);

  // center_two anywhere.
  const std::array<double, 3> center_two = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_two);

  // Place the flat disk centre 7–90% of radius_two below the sphere centre
  // in z, covering most of the 5–95% ASSERT range.  Keep x,y aligned with
  // center_two so that the x,y offset check is trivially satisfied.
  const double center_one_z =
      center_two[2] - radius_two * (0.07 + 0.83 * unit_dis(gen));
  const std::array<double, 3> center_one = {center_two[0], center_two[1],
                                            center_one_z};
  CAPTURE(center_one);

  // Projection point: 30–85% of radius_two above the sphere centre, covering
  // most of the dist_proj <= 0.95 * radius_two ASSERT range.
  const double proj_scale = 0.30 + 0.55 * unit_dis(gen);
  const std::array<double, 3> proj_center = {
      center_two[0], center_two[1], center_two[2] + proj_scale * radius_two};
  CAPTURE(proj_center);

  // z_sphere_extent: must be below center_one_z (ensuring t_sphere > 1) and
  // above center_two[2] - radius_two (staying on the sphere).  Compute the
  // available gap between center_one_z and the sphere bottom and choose a
  // fraction [0.1, 0.5] of it, guaranteeing the constraint holds even when
  // center_one_z is placed deep (close to 90% of radius_two below center).
  const double depth = center_two[2] - center_one_z;
  const double available = radius_two - depth;
  const double z_sphere_extent =
      center_one_z - available * (0.1 + 0.4 * unit_dis(gen));
  CAPTURE(z_sphere_extent);

  const CoordinateMaps::CylindricalFlatEndcapInterior map(
      center_one, center_two, proj_center, z_sphere_extent, radius_two);

  test_suite_for_map_on_cylinder(map, 0.0, 1.0);

  const Approx local_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);

  // Geometric check: the rim of the source cylinder at zbar=-1
  // (i.e. source point (cos phi, sin phi, -1)) must map to the circle at
  // z = z_sphere_extent on the outer sphere.
  for (size_t i = 0; i < 5; ++i) {
    const double phi = angle_dis(gen);
    const std::array<double, 3> rim_point = {std::cos(phi), std::sin(phi),
                                             -1.0};
    const auto rim_image = map(rim_point);
    // The image lies on the sphere of radius radius_two centred at center_two.
    CHECK(std::sqrt(square(rim_image[0] - center_two[0]) +
                    square(rim_image[1] - center_two[1]) +
                    square(rim_image[2] - center_two[2])) ==
          local_approx(radius_two));
    // The z-coordinate equals z_sphere_extent.
    CHECK(rim_image[2] == local_approx(z_sphere_extent));

    // Geometric check: the flat face (zbar=+1) lies at z = center_one[2].
    const double rho = unit_dis(gen);
    const std::array<double, 3> flat_point = {rho * std::cos(phi),
                                              rho * std::sin(phi), 1.0};
    const auto flat_image = map(flat_point);
    CHECK(flat_image[2] == local_approx(center_one[2]));
  }
}

// Test with a non-zero x,y offset between center_one and center_two to
// exercise the |C1 - C2| <= R1 + R2 ASSERT path.
void test_cylindrical_flat_endcap_interior_offset() {
  INFO("CylindricalFlatEndcapInterior with non-zero x,y offset");
  const std::array<double, 3> center_two = {0.0, 0.0, 0.0};
  const std::array<double, 3> center_one = {0.5, 0.3, -0.2};
  const std::array<double, 3> proj_center = {0.0, 0.0, 0.3};
  const double z_sphere_extent = -0.7;
  const double radius_two = 1.0;

  const CoordinateMaps::CylindricalFlatEndcapInterior map(
      center_one, center_two, proj_center, z_sphere_extent, radius_two);

  test_suite_for_map_on_cylinder(map, 0.0, 1.0);

  const Approx local_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);

  // Reconstruct the flat-disk radius the constructor computed internally.
  const double t_sphere_ref =
      (z_sphere_extent - proj_center[2]) / (center_one[2] - proj_center[2]);
  const double r_rim_ref =
      std::sqrt(square(radius_two) - square(z_sphere_extent - center_two[2]));
  const double radius_one = r_rim_ref / t_sphere_ref;

  for (size_t i = 0; i < 5; ++i) {
    const double phi = angle_dis(gen);
    const auto rim_image = map(std::array{std::cos(phi), std::sin(phi), -1.0});

    // Verify on sphere.
    CHECK(std::sqrt(square(rim_image[0] - center_two[0]) +
                    square(rim_image[1] - center_two[1]) +
                    square(rim_image[2] - center_two[2])) ==
          local_approx(radius_two));

    // Analytically compute the expected z from the focal projection:
    // disk point d = (R1*cos(phi)+C1_x, R1*sin(phi)+C1_y, C1_z)
    // ray from P through d: P + t*(d - P)
    // solve |P + t*(d-P) - C2|^2 = R2^2 for t (far root, t > 1).
    const double vx =
        radius_one * std::cos(phi) + center_one[0] - proj_center[0];
    const double vy =
        radius_one * std::sin(phi) + center_one[1] - proj_center[1];
    const double vz = center_one[2] - proj_center[2];
    const double ux = proj_center[0] - center_two[0];
    const double uy = proj_center[1] - center_two[1];
    const double uz = proj_center[2] - center_two[2];
    const double v2 = vx * vx + vy * vy + vz * vz;
    const double udotv = ux * vx + uy * vy + uz * vz;
    const double u2 = ux * ux + uy * uy + uz * uz;
    const double disc = udotv * udotv - v2 * (u2 - square(radius_two));
    // Far intersection (source_is_between_focus_and_target = true means sphere
    // is beyond the disk from P, so t > 1 and we take the + root).
    const double t_intersect = (-udotv + std::sqrt(disc)) / v2;
    const double z_expected = proj_center[2] + t_intersect * vz;
    CHECK(rim_image[2] == local_approx(z_expected));

    // Verify the flat face (zbar=+1) lies at z = center_one[2].
    const double rho = unit_dis(gen);
    const auto flat_image =
        map(std::array{rho * std::cos(phi), rho * std::sin(phi), 1.0});
    CHECK(flat_image[2] == local_approx(center_one[2]));
  }
}

// Test a shallow-disk geometry where the radius_two / radius_one ratio is
// large
void test_cylindrical_flat_endcap_interior_large_ratio() {
  INFO("CylindricalFlatEndcapInterior: large radius_two/radius_one ratio");
  const std::array<double, 3> center_two = {0.0, 0.0, 0.0};
  const std::array<double, 3> center_one = {0.0, 0.0, -0.1};
  const std::array<double, 3> proj_center = {0.0, 0.0, 0.0};

  const double z_sphere_extent = -0.625;
  const double radius_two = 1.0;

  const CoordinateMaps::CylindricalFlatEndcapInterior map(
      center_one, center_two, proj_center, z_sphere_extent, radius_two);

  test_suite_for_map_on_cylinder(map, 0.0, 1.0);
}

// Test a geometry where the projection point is far from the sphere center
void test_cylindrical_flat_endcap_interior_large_dist_proj() {
  INFO("CylindricalFlatEndcapInterior: large proj_center distance");
  const std::array<double, 3> center_two = {0.0, 0.0, 0.0};
  const std::array<double, 3> center_one = {0.0, 0.0, -0.5};
  const std::array<double, 3> proj_center = {0.0, 0.0, 0.85};

  const double z_sphere_extent = -0.7;
  const double radius_two = 1.0;

  const CoordinateMaps::CylindricalFlatEndcapInterior map(
      center_one, center_two, proj_center, z_sphere_extent, radius_two);

  test_suite_for_map_on_cylinder(map, 0.0, 1.0);
}

#ifdef SPECTRE_DEBUG
void test_assert_proj_center_inside_sphere() {
  INFO("FocallyLiftedMap asserts proj_center inside sphere");
  // proj_center at distance 1.5 is outside the sphere of radius 1.0.
  // impl_ is constructed before the CylindricalFlatEndcapInterior debug
  // asserts, so the FocallyLiftedMap assert fires first.
  CHECK_THROWS_WITH(
      (CoordinateMaps::CylindricalFlatEndcapInterior{
          {0.0, 0.0, -0.3}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.5}, -0.7, 1.0}),
      Catch::Matchers::ContainsSubstring(
          "proj_center must be strictly inside the sphere"));
}
#endif

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.CylindricalFlatEndcapInterior",
                  "[Domain][Unit]") {
  test_cylindrical_flat_endcap_interior();
  test_cylindrical_flat_endcap_interior_offset();
  test_cylindrical_flat_endcap_interior_large_ratio();
  test_cylindrical_flat_endcap_interior_large_dist_proj();
#ifdef SPECTRE_DEBUG
  test_assert_proj_center_inside_sphere();
#endif
  CHECK(not CoordinateMaps::CylindricalFlatEndcapInterior{}.is_identity());
}
}  // namespace domain
