// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <random>

#include "Domain/CoordinateMaps/CylindricalSide.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"

namespace domain {
namespace {
// If projection_point_near_z_plane is true, then the projection
// point is placed near or on one of the z_planes, and near the center
// of that plane.  Otherwise, the projection point is placed somewhere
// away from the z_planes.
//
// Also, for projection_point_near_z_plane=true, the allowed value of
// sphere_two is much larger than for projection_point_near_z_plane=false;
// this reflects our usage of the maps.
void test_cylindrical_side_sphere_two_encloses_sphere_one(
    const bool projection_point_near_z_plane) {
  INFO("CylindricalSideSphereTwoEnclosesSphereOne");
  // Set up random number generator
  MAKE_GENERATOR(gen);

  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random centers for sphere_one and sphere_two
  const std::array<double, 3> center_one = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_one);

  const auto center_two = [&projection_point_near_z_plane, &center_one,
                           &interval_dis,
                           &gen]() noexcept -> std::array<double, 3> {
    if (projection_point_near_z_plane) {
      return {center_one[0], center_one[1], interval_dis(gen)};
    } else {
      return {interval_dis(gen), interval_dis(gen), interval_dis(gen)};
    }
  }();
  CAPTURE(center_two);
  const double dist_between_spheres =
      sqrt(square(center_two[0] - center_one[0]) +
           square(center_two[1] - center_one[1]) +
           square(center_two[2] - center_one[2]));

  // Pick radius of sphere_one not too small compared to the distance
  // between the centers.
  const double radius_one = projection_point_near_z_plane
                                ? dist_between_spheres + 0.1 + unit_dis(gen)
                                : 0.3 * dist_between_spheres + unit_dis(gen);
  CAPTURE(radius_one);

  // Now construct sphere_two which we make sure encloses sphere_one,
  // but doesn't have too large of a radius.
  const double radius_two =
      projection_point_near_z_plane
          ? (3.0 * unit_dis(gen) + 1.01) * (radius_one + dist_between_spheres)
          : (unit_dis(gen) + 1.01) * (radius_one + dist_between_spheres);
  CAPTURE(radius_two);

  const std::array<double, 2> z_planes =
      [&gen, &unit_dis, &center_one, &radius_one,
       &projection_point_near_z_plane]() noexcept {
        if (projection_point_near_z_plane) {
          // Make sure each z_plane intersects sphere_one in two locations;
          // to do this, ensure that each plane is no closer than 25% of the
          // radius to the min/max z-extents of sphere_one and no closer than
          // 2% of the radius to the center.
          const double z_plane_1 =
              center_one[2] - (0.02 + 0.73 * unit_dis(gen)) * radius_one;
          const double z_plane_2 =
              center_one[2] + (0.02 + 0.73 * unit_dis(gen)) * radius_one;
          return std::array<double, 2>{z_plane_1, z_plane_2};
        } else {
          // Make sure each z_plane intersects sphere_one in two locations;
          // to do this, ensure that each plane is no closer than 10% of the
          // radius to the min/max z-extents of sphere_one and no closer than
          // 2% of the radius to the center.
          const double z_plane_1 =
              center_one[2] - (0.02 + 0.88 * unit_dis(gen)) * radius_one;
          const double z_plane_2 =
              center_one[2] + (0.02 + 0.88 * unit_dis(gen)) * radius_one;
          return std::array<double, 2>{z_plane_1, z_plane_2};
        }
      }();
  CAPTURE(z_planes);

  // Keep proj_center inside sphere_1 and between (or on) the z_planes.
  const std::array<double, 3> proj_center =
      [&z_planes, &center_one, &radius_one, &gen, &unit_dis, &angle_dis,
       &projection_point_near_z_plane]() noexcept {
        // z is z coordinate of projection point.
        // rho_max_factor is rho_max/rho_sphere_one (see below for
        // definition of rho_sphere_one).
        const auto & [z, rho_max_factor] =
            [&z_planes, &projection_point_near_z_plane, &unit_dis,
             &gen]() noexcept -> std::array<double, 2> {
          if (projection_point_near_z_plane) {
            // Place z exactly at one of the z-planes.
            // Choose which plane randomly, with probability 1/2.
            const double z_local =
                unit_dis(gen) < 0.5 ? z_planes[0] : z_planes[1];
            // Also, set rho to something very small.
            return {z_local, 1.e-25};
          } else {
            // z is not near one of the z-planes.
            // Note that z <= z_min but z < z_max.
            const double z_min =
                z_planes[0] + 0.1 * (z_planes[1] - z_planes[0]);
            const double z_max =
                z_planes[1] - 0.1 * (z_planes[1] - z_planes[0]);
            const double z_local = z_min + (z_max - z_min) * unit_dis(gen);
            return {z_local, 0.85};
          }
        }();
        // rho_sphere_one is the radius of the circle formed by the
        // intersection of sphere_one and a plane at coordinate z.
        const double rho_sphere_one =
            radius_one * sqrt(1.0 - square((z - center_one[2]) / radius_one));
        const double rho_max = rho_sphere_one * rho_max_factor;
        const double rho = unit_dis(gen) * rho_max;
        const double phi = angle_dis(gen);
        return std::array<double, 3>{center_one[0] + rho * cos(phi),
                                     center_one[1] + rho * sin(phi), z};
      }();
  CAPTURE(proj_center);

  const CoordinateMaps::CylindricalSide map(center_one, center_two, proj_center,
                                            radius_one, radius_two, z_planes[0],
                                            z_planes[1]);
  test_suite_for_map_on_cylinder(map, 1.0, 2.0);
}

void test_cylindrical_side_sphere_one_encloses_sphere_two() {
  INFO("CylindricalSideSphereOneEnclosesSphereTwo");
  // Set up random number generator
  MAKE_GENERATOR(gen);

  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> interval_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2.0 * M_PI);

  // Choose some random center for sphere_one
  const std::array<double, 3> center_one = {
      interval_dis(gen), interval_dis(gen), interval_dis(gen)};
  CAPTURE(center_one);

  // Pick radius of sphere_one
  const double radius_one = 1.5 * (1.0 + unit_dis(gen));
  CAPTURE(radius_one);

  // Make sure each z_plane intersects sphere_one in two locations;
  // to do this, ensure that each plane is no closer than 8% of the
  // radius to the min/max z-extents of sphere_one.
  // Also make sure that each z_plane is not more than 20% away from
  // the center of sphere_one (because we need to fit sphere_two between
  // the planes).
  const std::array<double, 2> z_planes = [&gen, &unit_dis, &center_one,
                                          &radius_one]() noexcept {
    const double z_plane_1 =
        center_one[2] - (0.2 + 0.72 * unit_dis(gen)) * radius_one;
    const double z_plane_2 =
        center_one[2] + (0.2 + 0.72 * unit_dis(gen)) * radius_one;
    return std::array<double, 2>{z_plane_1, z_plane_2};
  }();
  CAPTURE(z_planes);

  // Choose sphere_two to be fully contained inside the z_planes,
  // and not too small.
  const double radius_two =
      (z_planes[1] - z_planes[0]) * 0.25 * (1.0 + unit_dis(gen));
  CAPTURE(radius_two);

  // Choose center_two so that sphere_two is contained inside sphere_one
  // and inside the z_planes.
  const std::array<double, 3> center_two = [&z_planes, &radius_one, &radius_two,
                                            &center_one, &angle_dis, &unit_dis,
                                            &gen]() noexcept {
    const double center_two_z_min = z_planes[0] + radius_two;
    const double center_two_z_max = z_planes[1] - radius_two;
    const double center_two_z =
        center_two_z_min +
        (center_two_z_max - center_two_z_min) * unit_dis(gen);
    // Choose 0.85 so that there is some space between sphere_one and
    // sphere_two.
    const double rho = 0.85 * unit_dis(gen) *
                       sqrt(square(radius_one - radius_two) -
                            square(center_one[2] - center_two_z));
    const double phi = angle_dis(gen);
    return std::array<double, 3>{center_one[0] + rho * cos(phi),
                                 center_one[1] + rho * sin(phi), center_two_z};
  }();
  CAPTURE(center_two);

  // Keep proj_center inside sphere_two.
  const std::array<double, 3> proj_center = [&center_two, &radius_two, &gen,
                                             &unit_dis, &angle_dis]() noexcept {
    // choose 0.9 so that proj_center is not on edge of sphere_two.
    const double r = 0.9 * radius_two * unit_dis(gen);
    const double theta = 2.0 * angle_dis(gen);
    const double phi = angle_dis(gen);
    return std::array<double, 3>{center_two[0] + r * sin(theta) * cos(phi),
                                 center_two[1] + r * sin(theta) * sin(phi),
                                 center_two[2] + r * cos(theta)};
  }();
  CAPTURE(proj_center);

  const CoordinateMaps::CylindricalSide map(center_one, center_two, proj_center,
                                            radius_one, radius_two, z_planes[0],
                                            z_planes[1]);
  test_suite_for_map_on_cylinder(map, 1.0, 2.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.CylindricalSide",
                  "[Domain][Unit]") {
  test_cylindrical_side_sphere_two_encloses_sphere_one(true);
  test_cylindrical_side_sphere_two_encloses_sphere_one(false);
  test_cylindrical_side_sphere_one_encloses_sphere_two();
  CHECK(not CoordinateMaps::CylindricalSide{}.is_identity());
}
}  // namespace domain
