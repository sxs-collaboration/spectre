// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <pup.h>
#include <random>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {
namespace {
void test_suite_for_frustum(const bool with_equiangular_map) {
  INFO("Suite for frustum");
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> lower_bound_lower_base_dis(-7, -3);
  std::uniform_real_distribution<> upper_bound_lower_base_dis(3, 7);
  std::uniform_real_distribution<> lower_bound_upper_base_dis(-13.5, -9);
  std::uniform_real_distribution<> upper_bound_upper_base_dis(9, 13.5);
  std::uniform_real_distribution<> angle_dis(55.0, 125.0);

  const double lower_x_lower_base = lower_bound_lower_base_dis(gen);
  CAPTURE(lower_x_lower_base);
  const double lower_y_lower_base = lower_bound_lower_base_dis(gen);
  CAPTURE(lower_y_lower_base);
  const double upper_x_lower_base = upper_bound_lower_base_dis(gen);
  CAPTURE(upper_x_lower_base);
  const double upper_y_lower_base = upper_bound_lower_base_dis(gen);
  CAPTURE(upper_y_lower_base);
  const double lower_x_upper_base = lower_bound_upper_base_dis(gen);
  CAPTURE(lower_x_upper_base);
  const double lower_y_upper_base = lower_bound_upper_base_dis(gen);
  CAPTURE(lower_y_upper_base);
  const double upper_x_upper_base = upper_bound_upper_base_dis(gen);
  CAPTURE(upper_x_upper_base);
  const double upper_y_upper_base = upper_bound_upper_base_dis(gen);
  CAPTURE(upper_y_upper_base);
  const double upper_z = 3.0;
  CAPTURE(upper_z);
  const double lower_z = -1.0;
  CAPTURE(lower_z);
  const double opening_angle = angle_dis(gen) * M_PI / 180.0;
  CAPTURE(opening_angle * 180.0 / M_PI);

  // For diagnostic purposes, compute other Frustum quantities:
  // Frustum cross factor is small when the diagonals across each of the bases
  // are aligned, and large when there is an angle between them.
  const double frustum_cross = (upper_x_lower_base - lower_x_lower_base) *
                                   (upper_y_upper_base - lower_y_upper_base) -
                               (upper_y_lower_base - lower_y_lower_base) *
                                   (upper_x_upper_base - lower_x_upper_base);
  CAPTURE(frustum_cross);

  // Computes how vertical the walls are of the frustum. When these
  // quantities are zero, the frustum walls are parallel to the z-axis. This
  // is undesired because when bulging out the Frustum the coordinates are
  // moved along radial lines emanating from the origin.
  const double x_wall_sep_upper = upper_x_upper_base - upper_x_lower_base;
  const double x_wall_sep_lower = lower_x_upper_base - lower_x_lower_base;
  const double y_wall_sep_upper = upper_y_upper_base - upper_y_lower_base;
  const double y_wall_sep_lower = lower_y_upper_base - lower_y_lower_base;
  const double min_wall_sep = std::min(
      {x_wall_sep_upper, x_wall_sep_lower, y_wall_sep_upper, y_wall_sep_lower});
  CAPTURE(x_wall_sep_upper);
  CAPTURE(x_wall_sep_lower);
  CAPTURE(y_wall_sep_upper);
  CAPTURE(y_wall_sep_lower);
  CAPTURE(min_wall_sep);

  // Computes how far away the base of the Frustum is from the origin. Since
  // the bulged Frustum is constructed by moving coordinates along radial lines,
  // having the base of the frustum be shifted away from the origin will lead to
  // a distorted shape. If the Frustum test fails, check if the com_shift is
  // above 1.8. A test failure might occur in 1/25,000 cases.
  const double x_com_bottom = 0.5 * (lower_x_lower_base + upper_x_lower_base);
  const double y_com_bottom = 0.5 * (lower_y_lower_base + upper_y_lower_base);
  const double com_shift = sqrt(square(x_com_bottom) + square(y_com_bottom));
  CAPTURE(com_shift);

  for (OrientationMapIterator<3> map_i{}; map_i; ++map_i) {
    if (get(determinant(discrete_rotation_jacobian(*map_i))) < 0.0) {
      continue;
    }
    const std::array<std::array<double, 2>, 4> face_vertices{
        {{{lower_x_lower_base, lower_y_lower_base}},
         {{upper_x_lower_base, upper_y_lower_base}},
         {{lower_x_upper_base, lower_y_upper_base}},
         {{upper_x_upper_base, upper_y_upper_base}}}};

    // The parameters of the Frustum are chosen so that the angles between
    // the faces are not too obtuse or acute. The ratio of the length
    // of the longer base to the height of the Frustum is at most 7:1. Frustums
    // with larger ratios cannot be bulged out without the root find beginning
    // to fail in extreme cases.
    const CoordinateMaps::Frustum frustum_map(
        face_vertices, lower_z, upper_z, map_i(), with_equiangular_map, 1.2,
        false, 1.0, 1.0, opening_angle);
    test_suite_for_map_on_unit_cube(frustum_map);
  }
}

void test_frustum_fail() {
  INFO("Frustum fail");
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const CoordinateMaps::Frustum map(face_vertices, 2.0, 5.0,
                                    OrientationMap<3>{});

  // For the choice of params above, any point with z<=-1 should fail.
  const std::array<double, 3> test_mapped_point1{{3.0, 3.0, -1.0}};
  const std::array<double, 3> test_mapped_point2{{6.0, -7.0, -1.0}};
  const std::array<double, 3> test_mapped_point3{{6.0, -7.0, -3.0}};

  // This is outside the mapped frustum, so inverse should either
  // return the correct inverse (which happens to be computable for
  // this point) or it should return nullopt.
  const std::array<double, 3> test_mapped_point4{{0.0, 0.0, 9.0}};

  CHECK_FALSE(map.inverse(test_mapped_point1).has_value());
  CHECK_FALSE(map.inverse(test_mapped_point2).has_value());
  CHECK_FALSE(map.inverse(test_mapped_point3).has_value());
  if (map.inverse(test_mapped_point4).has_value()) {
    CHECK_ITERABLE_APPROX(map(map.inverse(test_mapped_point4).value()),
                          test_mapped_point4);
  }
}

void test_alignment() {
  INFO("Alignment");
  // This test tests that the logical axes point along the expected directions
  // in physical space

  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const double lower_bound = 2.0;
  const double upper_bound = 5.0;

  const auto wedge_directions = all_wedge_directions();
  const CoordinateMaps::Frustum map_upper_zeta(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[0]);  // Upper Z frustum
  const CoordinateMaps::Frustum map_upper_eta(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[2]);  // Upper Y frustum
  const CoordinateMaps::Frustum map_upper_xi(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[4]);  // Upper X Frustum
  const CoordinateMaps::Frustum map_lower_zeta(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[1]);  // Lower Z frustum
  const CoordinateMaps::Frustum map_lower_eta(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[3]);  // Lower Y frustum
  const CoordinateMaps::Frustum map_lower_xi(
      face_vertices, lower_bound, upper_bound,
      wedge_directions[5]);  // Lower X frustum
  const std::array<double, 3> lowest_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> along_xi{{1.0, -1.0, -1.0}};
  const std::array<double, 3> along_eta{{-1.0, 1.0, -1.0}};
  const std::array<double, 3> along_zeta{{-1.0, -1.0, 1.0}};

  const std::array<double, 3> lowest_physical_corner_in_map_upper_zeta{
      {-2.0, -2.0, 2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_upper_eta{
      {-2.0, 2.0, -2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_upper_xi{
      {2.0, -2.0, -2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_lower_zeta{
      {-2.0, 2.0, -2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_lower_eta{
      {-2.0, -2.0, 2.0}};
  const std::array<double, 3> lowest_physical_corner_in_map_lower_xi{
      {-2.0, 2.0, -2.0}};

  // Test that this map's logical axes point along +X, +Y, +Z:
  CHECK(map_upper_zeta(along_xi)[0] == approx(2.0));
  CHECK(map_upper_zeta(along_eta)[1] == approx(2.0));
  CHECK(map_upper_zeta(along_zeta)[2] == approx(5.0));
  CHECK_ITERABLE_APPROX(map_upper_zeta(lowest_corner),
                        lowest_physical_corner_in_map_upper_zeta);

  // Test that this map's logical axes point along +Z, +X, +Y:
  CHECK(map_upper_eta(along_xi)[2] == approx(2.0));
  CHECK(map_upper_eta(along_eta)[0] == approx(2.0));
  CHECK(map_upper_eta(along_zeta)[1] == approx(5.0));
  CHECK_ITERABLE_APPROX(map_upper_eta(lowest_corner),
                        lowest_physical_corner_in_map_upper_eta);

  // Test that this map's logical axes point along +Y, +Z, +X:
  CHECK(map_upper_xi(along_xi)[1] == approx(2.0));
  CHECK(map_upper_xi(along_eta)[2] == approx(2.0));
  CHECK(map_upper_xi(along_zeta)[0] == approx(5.0));
  CHECK_ITERABLE_APPROX(map_upper_xi(lowest_corner),
                        lowest_physical_corner_in_map_upper_xi);

  // Test that this map's logical axes point along +X, -Y, -Z:
  CHECK(map_lower_zeta(along_xi)[0] == approx(2.0));
  CHECK(map_lower_zeta(along_eta)[1] == approx(-2.0));
  CHECK(map_lower_zeta(along_zeta)[2] == approx(-5.0));
  CHECK_ITERABLE_APPROX(map_lower_zeta(lowest_corner),
                        lowest_physical_corner_in_map_lower_zeta);

  // Test that this map's logical axes point along -Z, +X, -Y:
  CHECK(map_lower_eta(along_xi)[2] == approx(-2.0));
  CHECK(map_lower_eta(along_eta)[0] == approx(2.0));
  CHECK(map_lower_eta(along_zeta)[1] == approx(-5.0));
  CHECK_ITERABLE_APPROX(map_lower_eta(lowest_corner),
                        lowest_physical_corner_in_map_lower_eta);

  // Test that this map's logical axes point along -Y, +Z, -X:
  CHECK(map_lower_xi(along_xi)[1] == approx(-2.0));
  CHECK(map_lower_xi(along_eta)[2] == approx(2.0));
  CHECK(map_lower_xi(along_zeta)[0] == approx(-5.0));
  CHECK_ITERABLE_APPROX(map_lower_xi(lowest_corner),
                        lowest_physical_corner_in_map_lower_xi);
}

void test_auto_projective_scale_factor() {
  INFO("Auto projective scale factor");
  // This test tests that the correct suggested projective scale factor
  // is computed based on the side lengths of the frustum.

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> lower_bound_dis(-14, -2);
  std::uniform_real_distribution<> upper_bound_dis(2, 14);
  std::uniform_real_distribution<> logical_dis(-1, 1);

  const double lower_x_lower_base = lower_bound_dis(gen);
  CAPTURE(lower_x_lower_base);
  const double lower_y_lower_base = lower_bound_dis(gen);
  CAPTURE(lower_y_lower_base);
  const double upper_x_lower_base = upper_bound_dis(gen);
  CAPTURE(upper_x_lower_base);
  const double upper_y_lower_base = upper_bound_dis(gen);
  CAPTURE(upper_y_lower_base);
  const double lower_x_upper_base = lower_bound_dis(gen);
  CAPTURE(lower_x_upper_base);
  const double lower_y_upper_base = lower_bound_dis(gen);
  CAPTURE(lower_y_upper_base);
  const double upper_x_upper_base = upper_bound_dis(gen);
  CAPTURE(upper_x_upper_base);
  const double upper_y_upper_base = upper_bound_dis(gen);
  CAPTURE(upper_y_upper_base);
  const double xi = logical_dis(gen);
  CAPTURE(xi);
  const double eta = logical_dis(gen);
  CAPTURE(eta);
  const double zeta = logical_dis(gen);
  CAPTURE(zeta);

  const std::array<double, 3> logical_coord{{0.0, 0.0, zeta}};
  const double lower_bound = 2.0;
  const double upper_bound = 5.0;
  const double sigma_z = 0.5 * (upper_bound + lower_bound);
  const double delta_z = 0.5 * (upper_bound - lower_bound);
  const double w_delta = sqrt(((upper_x_lower_base - lower_x_lower_base) *
                               (upper_y_lower_base - lower_y_lower_base)) /
                              ((upper_x_upper_base - lower_x_upper_base) *
                               (upper_y_upper_base - lower_y_upper_base)));
  const double projective_zeta = (w_delta - 1.0 + zeta * (w_delta + 1.0)) /
                                 (w_delta + 1.0 + zeta * (w_delta - 1.0));
  const double expected_physical_z = sigma_z + delta_z * (projective_zeta);
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{lower_x_lower_base, lower_y_lower_base}},
       {{upper_x_lower_base, upper_y_lower_base}},
       {{lower_x_upper_base, lower_y_upper_base}},
       {{upper_x_upper_base, upper_y_upper_base}}}};

  const CoordinateMaps::Frustum map(face_vertices, lower_bound, upper_bound, {},
                                    false, 1.0, true);  // Upper Z frustum
  CHECK(map(logical_coord)[2] == approx(expected_physical_z));
}

void test_is_identity() {
  INFO("Is identity");
  check_if_map_is_identity(CoordinateMaps::Frustum{
      std::array<std::array<double, 2>, 4>{
          {{{-1.0, -1.0}}, {{1.0, 1.0}}, {{-1.0, -1.0}}, {{1.0, 1.0}}}},
      -1.0, 1.0, OrientationMap<3>{}, false, 1.0});
  CHECK(not CoordinateMaps::Frustum{
      std::array<std::array<double, 2>, 4>{
          {{{-1.0, -1.0}}, {{2.0, 1.0}}, {{-1.0, -3.0}}, {{1.0, 1.0}}}},
      -1.0, 1.0, OrientationMap<3>{}, false, 1.5}
                .is_identity());
}

void test_bulged_frustum_jacobian() {
  INFO("Bulged frustum jacobian");
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const CoordinateMaps::Frustum map(
      face_vertices, 2.0, 5.0, OrientationMap<3>{}, false, 1.0, false, 1.0);

  const std::array<double, 3> test_point1{{-1.0, 0.25, 0.0}};
  const std::array<double, 3> test_point2{{1.0, 1.0, -0.5}};
  const std::array<double, 3> test_point3{{0.7, -0.2, 0.4}};
  const std::array<double, 3> test_point4{{0.0, 0.0, 0.0}};

  test_jacobian(map, test_point1);
  test_jacobian(map, test_point2);
  test_jacobian(map, test_point3);
  test_jacobian(map, test_point4);
}

void test_bulged_frustum_inv_jacobian() {
  INFO("Bulged frustum inverse jacobian");
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const CoordinateMaps::Frustum map(
      face_vertices, 2.0, 5.0, OrientationMap<3>{}, false, 1.0, false, 1.0);

  const std::array<double, 3> test_point1{{-1.0, 0.25, 0.0}};
  const std::array<double, 3> test_point2{{1.0, 1.0, -0.5}};
  const std::array<double, 3> test_point3{{0.7, -0.2, 0.4}};
  const std::array<double, 3> test_point4{{0.0, 0.0, 0.0}};

  test_inv_jacobian(map, test_point1);
  test_inv_jacobian(map, test_point2);
  test_inv_jacobian(map, test_point3);
  test_inv_jacobian(map, test_point4);
}

void test_bulged_frustum_inv_map() {
  INFO("Bulged frustum inverse map");
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const CoordinateMaps::Frustum map(
      face_vertices, 2.0, 5.0, OrientationMap<3>{}, false, 1.0, false, 1.0);

  const std::array<double, 3> test_point1{{-1.0, 0.25, 0.0}};
  const std::array<double, 3> test_point2{{1.0, 1.0, -0.5}};
  const std::array<double, 3> test_point3{{0.7, -0.2, 0.4}};
  const std::array<double, 3> test_point4{{0.0, 0.0, 0.01}};

  test_inverse_map(map, test_point1);
  test_inverse_map(map, test_point2);
  test_inverse_map(map, test_point3);
  test_inverse_map(map, test_point4);
}

void test_bulged_frustum_equiangular_full() {
  INFO("Bulged frustum jacobian");
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const CoordinateMaps::Frustum map(
      face_vertices, 2.0, 4.0, OrientationMap<3>{}, true, 1.0, false, 1.0, 0.0);

  // Points on the upper +zeta face:
  const std::array<std::array<double, 3>, 4> cornerpts{{{{-1.0, -1.0, 1.0}},
                                                        {{-1.0, 1.0, 1.0}},
                                                        {{1.0, -1.0, 1.0}},
                                                        {{1.0, 1.0, 1.0}}}};

  const std::array<std::array<double, 3>, 3> zeroxipts{
      {{{0.0, -1.0, 1.0}}, {{0.0, 0.0, 1.0}}, {{0.0, 1.0, 1.0}}}};

  const std::array<std::array<double, 3>, 3> zeroetapts{
      {{{-1.0, 0.0, 1.0}}, {{0.0, 0.0, 1.0}}, {{1.0, 0.0, 1.0}}}};

  // If Frustum is successfully bulged, points on upper +zeta face
  // should have the same radius:
  const double radius = 4.0 * sqrt(3.0);
  for (size_t i = 0; i < 3; i++) {
    CHECK(magnitude(map(gsl::at(cornerpts, i))) == approx(radius));
    CHECK(magnitude(map(gsl::at(zeroxipts, i))) == approx(radius));
    CHECK(magnitude(map(gsl::at(zeroetapts, i))) == approx(radius));
  }
  CHECK(magnitude(map(gsl::at(cornerpts, 3))) == approx(radius));

  // If Frustum is successfully equiangular, distances equally spaced
  // in logical space correspond to equal angular distances. For a map
  // where a logical axis subtends an angle of theta, the jacobian
  // is radius * theta / 2 at any point along this axis.
  const double radius_times_theta_over_two = radius * M_PI_4;
  for (size_t i = 0; i < 3; i++) {
    const std::array<double, 3> zeroeta_dx_dxi{
        {map.jacobian(gsl::at(zeroetapts, i)).get(0, 0),
         map.jacobian(gsl::at(zeroetapts, i)).get(1, 0),
         map.jacobian(gsl::at(zeroetapts, i)).get(2, 0)}};
    CHECK(magnitude(zeroeta_dx_dxi) == approx(radius_times_theta_over_two));

    const std::array<double, 3> zeroxi_dx_deta{
        {map.jacobian(gsl::at(zeroxipts, i)).get(0, 1),
         map.jacobian(gsl::at(zeroxipts, i)).get(1, 1),
         map.jacobian(gsl::at(zeroxipts, i)).get(2, 1)}};
    CHECK(magnitude(zeroxi_dx_deta) == approx(radius_times_theta_over_two));
  }
}

void test_bulged_frustum_equiangular_upper() {
  INFO("Bulged frustum jacobian");
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{0.0, -2.0}}, {{2.0, 2.0}}, {{0.0, -4.0}}, {{4.0, 4.0}}}};
  const CoordinateMaps::Frustum map(
      face_vertices, 2.0, 4.0, OrientationMap<3>{}, true, 1.0, false, 1.0, 1.0);

  const std::array<std::array<double, 3>, 4> cornerpts{{{{-1.0, -1.0, 1.0}},
                                                        {{-1.0, 1.0, 1.0}},
                                                        {{1.0, -1.0, 1.0}},
                                                        {{1.0, 1.0, 1.0}}}};

  const std::array<std::array<double, 3>, 3> lowerxipts{
      {{{-1.0, -1.0, 1.0}}, {{-1.0, 0.0, 1.0}}, {{-1.0, 1.0, 1.0}}}};

  const std::array<std::array<double, 3>, 3> zeroetapts{
      {{{-1.0, 0.0, 1.0}}, {{0.0, 0.0, 1.0}}, {{1.0, 0.0, 1.0}}}};

  // If Frustum is successfully bulged, points on upper +zeta face
  // should have the same radius:
  const double radius = 4.0 * sqrt(3.0);
  for (size_t i = 0; i < 3; i++) {
    CHECK(magnitude(map(gsl::at(cornerpts, i))) == approx(radius));
    CHECK(magnitude(map(gsl::at(lowerxipts, i))) == approx(radius));
    CHECK(magnitude(map(gsl::at(zeroetapts, i))) == approx(radius));
  }
  CHECK(magnitude(map(gsl::at(cornerpts, 3))) == approx(radius));

  // If Frustum is successfully equiangular, distances equally spaced
  // in logical space correspond to equal angular distances. For a map
  // where a logical axis subtends an angle of theta, the jacobian
  // is radius * theta / 2 at any point along this axis. For a full
  // wedge map, which this frustum emulates, the angle theta is pi/2,
  // as four congruent wedges will cover a great circle. The angle is
  // divided by two to account for the fact that the logical cube has a
  // length of two.
  const double radius_times_theta_over_two = radius * M_PI_4;
  for (size_t i = 0; i < 3; i++) {
    const std::array<double, 3> zeroeta_dx_dxi{
        {map.jacobian(gsl::at(zeroetapts, i)).get(0, 0),
         map.jacobian(gsl::at(zeroetapts, i)).get(1, 0),
         map.jacobian(gsl::at(zeroetapts, i)).get(2, 0)}};

    // The map only subtends half the angle in the xi direction when the
    // half equiangular map is used.
    CHECK(magnitude(zeroeta_dx_dxi) ==
          approx(0.5 * radius_times_theta_over_two));

    const std::array<double, 3> lowerxi_dx_deta{
        {map.jacobian(gsl::at(lowerxipts, i)).get(0, 1),
         map.jacobian(gsl::at(lowerxipts, i)).get(1, 1),
         map.jacobian(gsl::at(lowerxipts, i)).get(2, 1)}};

    // The map must be checked at the xi=-1 points as the lower xi points
    // on the upper half map correspond to the xi=0 points on the full half.
    CHECK(magnitude(lowerxi_dx_deta) == approx(radius_times_theta_over_two));
  }
}

void test_bulged_frustum_equiangular_lower() {
  INFO("Bulged frustum jacobian");
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{0.0, 2.0}}, {{-4.0, -4.0}}, {{0.0, 4.0}}}};
  const CoordinateMaps::Frustum map(face_vertices, 2.0, 4.0,
                                    OrientationMap<3>{}, true, 1.0, false, 1.0,
                                    -1.0);

  const std::array<std::array<double, 3>, 4> cornerpts{{{{-1.0, -1.0, 1.0}},
                                                        {{-1.0, 1.0, 1.0}},
                                                        {{1.0, -1.0, 1.0}},
                                                        {{1.0, 1.0, 1.0}}}};

  const std::array<std::array<double, 3>, 3> upperxipts{
      {{{1.0, -1.0, 1.0}}, {{1.0, 0.0, 1.0}}, {{1.0, 1.0, 1.0}}}};

  const std::array<std::array<double, 3>, 3> zeroetapts{
      {{{-1.0, 0.0, 1.0}}, {{0.0, 0.0, 1.0}}, {{1.0, 0.0, 1.0}}}};

  // If Frustum is successfully bulged, points on upper +zeta face
  // should have the same radius:
  const double radius = 4.0 * sqrt(3.0);
  for (size_t i = 0; i < 3; i++) {
    CHECK(magnitude(map(gsl::at(cornerpts, i))) == approx(radius));
    CHECK(magnitude(map(gsl::at(upperxipts, i))) == approx(radius));
    CHECK(magnitude(map(gsl::at(zeroetapts, i))) == approx(radius));
  }
  CHECK(magnitude(map(gsl::at(cornerpts, 3))) == approx(radius));

  // If Frustum is successfully equiangular, distances equally spaced
  // in logical space correspond to equal angular distances. For a map
  // where a logical axis subtends an angle of theta, the jacobian
  // is radius * theta / 2 at any point along this axis. For a full
  // wedge map, which this frustum emulates, the angle theta is pi/2,
  // as four congruent wedges will cover a great circle. The angle is
  // divided by two to account for the fact that the logical cube has a
  // length of two.
  const double radius_times_theta_over_two = radius * M_PI_4;
  for (size_t i = 0; i < 3; i++) {
    const std::array<double, 3> zeroeta_dx_dxi{
        {map.jacobian(gsl::at(zeroetapts, i)).get(0, 0),
         map.jacobian(gsl::at(zeroetapts, i)).get(1, 0),
         map.jacobian(gsl::at(zeroetapts, i)).get(2, 0)}};

    // The map only subtends half the angle in the xi direction when the
    // half equiangular map is used.
    CHECK(magnitude(zeroeta_dx_dxi) ==
          approx(0.5 * radius_times_theta_over_two));

    const std::array<double, 3> upperxi_dx_deta{
        {map.jacobian(gsl::at(upperxipts, i)).get(0, 1),
         map.jacobian(gsl::at(upperxipts, i)).get(1, 1),
         map.jacobian(gsl::at(upperxipts, i)).get(2, 1)}};

    // The map must be checked at the xi=+1 points as the upper xi points
    // on the lower half map correspond to the xi=0 points on the full half.
    CHECK(magnitude(upperxi_dx_deta) == approx(radius_times_theta_over_two));
  }
}

void test_frustum_fail_equiangular() {
  INFO("Frustum fail");
  const std::array<std::array<double, 2>, 4> face_vertices{
      {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
  const std::array<double, 3> test_mapped_point1{{0.0, 50.0, 9.0}};
  const std::array<double, 3> test_mapped_point2{{4.0, 4.0, 1.0}};
  const std::array<double, 3> test_mapped_point3{{3.0, 3.0, 1.99}};
  const CoordinateMaps::Frustum equiangular_map(
      face_vertices, 2.0, 5.0, OrientationMap<3>{}, true, 1.0, false, 1.0, 1.0);
  CHECK(not equiangular_map.inverse(test_mapped_point1).has_value());
  CHECK(not equiangular_map.inverse(test_mapped_point2).has_value());
  CHECK(not equiangular_map.inverse(test_mapped_point3).has_value());
}

void test_bulged_frustum_inverse() {
  // Check that the bulged frustum inverse map works with input coordinates
  // outside of the domain, and returns nullopt. The map involves a numerical
  // rootfind, which has to terminate cleanly for this to work. These parameters
  // are taken from a BBH domain where this case failed.
  CoordinateMaps::Frustum frustum{
      {{{{-40., -20.}},
        {{0., 20.}},
        {{-69.282032302755098385, -69.282032302755098385}},
        {{0., 69.282032302755098385}}}},
      20.,
      69.282032302755098385,
      OrientationMap<3>{{{Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                          Direction<3>::upper_eta()}}},
      true,
      0.288675,
      false,
      1.,
      -1.};
  const auto inverse =
      frustum.inverse({{-83.565015846289398382, -20.891253961572349596,
                        -82.337943622612172589}});
  CHECK_FALSE(inverse.has_value());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Frustum", "[Domain][Unit]") {
  test_frustum_fail();
  test_suite_for_frustum(false);  // Equidistant
  test_suite_for_frustum(true);   // Equiangular
  test_alignment();
  test_auto_projective_scale_factor();
  test_is_identity();
  test_bulged_frustum_jacobian();
  test_bulged_frustum_inv_jacobian();
  test_bulged_frustum_inv_map();
  test_bulged_frustum_equiangular_full();
  test_bulged_frustum_equiangular_upper();
  test_bulged_frustum_equiangular_lower();
  test_frustum_fail_equiangular();
  test_bulged_frustum_inverse();

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        const std::array<std::array<double, 2>, 4> face_vertices{
            {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
        const double lower_bound = 2.0;
        const double upper_bound = 5.0;
        const double projective_scale_factor = 0.0;
        const bool with_equiangular_map = false;

        auto failed_frustum = CoordinateMaps::Frustum(
            face_vertices, lower_bound, upper_bound, OrientationMap<3>{},
            with_equiangular_map, projective_scale_factor);
        static_cast<void>(failed_frustum);
      }()),
      Catch::Contains(
          "A projective scale factor of zero maps all coordinates to zero! Set "
          "projective_scale_factor to unity to turn off projective scaling."));

  CHECK_THROWS_WITH(
      ([]() {
        const std::array<std::array<double, 2>, 4> face_vertices{
            {{{-2.0, -2.0}}, {{-3.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
        const double lower_bound = 2.0;
        const double upper_bound = 5.0;

        auto failed_frustum = CoordinateMaps::Frustum(
            face_vertices, lower_bound, upper_bound, OrientationMap<3>{});
        static_cast<void>(failed_frustum);
      }()),
      Catch::Contains("The lower bound for a coordinate must be numerically "
                      "less than the upper bound for that coordinate."));

  CHECK_THROWS_WITH(
      ([]() {
        const std::array<std::array<double, 2>, 4> face_vertices{
            {{{-2.0, -2.0}}, {{2.0, -3.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
        const double lower_bound = 2.0;
        const double upper_bound = 5.0;

        auto failed_frustum = CoordinateMaps::Frustum(
            face_vertices, lower_bound, upper_bound, OrientationMap<3>{});
        static_cast<void>(failed_frustum);
      }()),
      Catch::Contains("The lower bound for a coordinate must be numerically "
                      "less than the upper bound for that coordinate."));

  CHECK_THROWS_WITH(
      ([]() {
        const std::array<std::array<double, 2>, 4> face_vertices{
            {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{-5.0, 4.0}}}};
        const double lower_bound = 2.0;
        const double upper_bound = 5.0;

        auto failed_frustum = CoordinateMaps::Frustum(
            face_vertices, lower_bound, upper_bound, OrientationMap<3>{});
        static_cast<void>(failed_frustum);
      }()),
      Catch::Contains("The lower bound for a coordinate must be numerically "
                      "less than the upper bound for that coordinate."));

  CHECK_THROWS_WITH(
      ([]() {
        const std::array<std::array<double, 2>, 4> face_vertices{
            {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, -5.0}}}};
        const double lower_bound = 2.0;
        const double upper_bound = 5.0;

        auto failed_frustum = CoordinateMaps::Frustum(
            face_vertices, lower_bound, upper_bound, OrientationMap<3>{});
        static_cast<void>(failed_frustum);
      }()),
      Catch::Contains("The lower bound for a coordinate must be numerically "
                      "less than the upper bound for that coordinate."));

  CHECK_THROWS_WITH(
      ([]() {
        const std::array<std::array<double, 2>, 4> face_vertices{
            {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
        const double lower_bound = 2.0;
        const double upper_bound = -2.0;

        auto failed_frustum = CoordinateMaps::Frustum(
            face_vertices, lower_bound, upper_bound, OrientationMap<3>{});
        static_cast<void>(failed_frustum);
      }()),
      Catch::Contains("The lower bound for a coordinate must be numerically "
                      "less than the upper bound for that coordinate."));

  CHECK_THROWS_WITH(
      ([]() {
        const std::array<std::array<double, 2>, 4> face_vertices{
            {{{-2.0, -2.0}}, {{2.0, 2.0}}, {{-4.0, -4.0}}, {{4.0, 4.0}}}};
        const double lower_bound = 2.0;
        const double upper_bound = 5.0;
        const double projective_scale_factor = 1.0;
        const bool with_equiangular_map = false;
        const double sphericity = 1.3;

        auto failed_frustum = CoordinateMaps::Frustum(
            face_vertices, lower_bound, upper_bound, OrientationMap<3>{},
            with_equiangular_map, projective_scale_factor, false, sphericity);
        static_cast<void>(failed_frustum);
      }()),
      Catch::Contains(
          "The sphericity must be set between 0.0, corresponding to a flat "
          "surface, and 1.0, corresponding to a spherical surface, inclusive. "
          "It is currently set to 1.3"));
#endif
}
}  // namespace domain
