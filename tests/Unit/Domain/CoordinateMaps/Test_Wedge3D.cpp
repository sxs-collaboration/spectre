// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <random>

#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace {
void test_wedge3d_all_directions(const bool with_equiangular_map) {
  // Set up random number generator
  const auto seed = std::random_device{}();
  std::mt19937 gen(seed);
  INFO("seed = " << seed);
  std::uniform_real_distribution<> unit_dis(0, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(5.2, 7);
  const double inner_radius = inner_dis(gen);
  CAPTURE_PRECISE(inner_radius);
  const double outer_radius = outer_dis(gen);
  CAPTURE_PRECISE(outer_radius);
  const double inner_sphericity = unit_dis(gen);
  CAPTURE_PRECISE(inner_sphericity);
  const double outer_sphericity = unit_dis(gen);
  CAPTURE_PRECISE(outer_sphericity);

  using WedgeHalves = CoordinateMaps::Wedge3D::WedgeHalves;
  const std::array<WedgeHalves, 3> halves_array = {
      {WedgeHalves::UpperOnly, WedgeHalves::LowerOnly, WedgeHalves::Both}};
  for (const auto& halves : halves_array) {
    for (const auto& direction : all_wedge_directions()) {
      const CoordinateMaps::Wedge3D wedge_map(
          inner_radius, outer_radius, direction, inner_sphericity,
          outer_sphericity, with_equiangular_map, halves);
      test_suite_for_map(wedge_map);
    }
  }
}

void test_wedge3d_alignment(const bool with_equiangular_map) {
  // This test tests that the logical axes point along the expected directions
  // in physical space

  const double inner_r = sqrt(3.0);
  const double outer_r = 2.0 * sqrt(3.0);

  const auto wedge_directions = all_wedge_directions();
  const CoordinateMaps::Wedge3D map_upper_zeta(
      inner_r, outer_r, wedge_directions[0], 0.0, 1.0,
      with_equiangular_map);  // Upper Z wedge
  const CoordinateMaps::Wedge3D map_upper_eta(
      inner_r, outer_r, wedge_directions[2], 0.0, 1.0,
      with_equiangular_map);  // Upper Y wedge
  const CoordinateMaps::Wedge3D map_upper_xi(
      inner_r, outer_r, wedge_directions[4], 0.0, 1.0,
      with_equiangular_map);  // Upper X Wedge
  const CoordinateMaps::Wedge3D map_lower_zeta(
      inner_r, outer_r, wedge_directions[1], 0.0, 1.0,
      with_equiangular_map);  // Lower Z wedge
  const CoordinateMaps::Wedge3D map_lower_eta(
      inner_r, outer_r, wedge_directions[3], 0.0, 1.0,
      with_equiangular_map);  // Lower Y wedge
  const CoordinateMaps::Wedge3D map_lower_xi(
      inner_r, outer_r, wedge_directions[5], 0.0, 1.0,
      with_equiangular_map);  // Lower X wedge
  const std::array<double, 3> lowest_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> along_xi{{1.0, -1.0, -1.0}};
  const std::array<double, 3> along_eta{{-1.0, 1.0, -1.0}};
  const std::array<double, 3> along_zeta{{-1.0, -1.0, 1.0}};

  const std::array<double, 3> lowest_physical_corner_upper_zeta{
      {-1.0, -1.0, 1.0}};
  const std::array<double, 3> lowest_physical_corner_upper_eta{
      {-1.0, 1.0, -1.0}};
  const std::array<double, 3> lowest_physical_corner_upper_xi{
      {1.0, -1.0, -1.0}};
  const std::array<double, 3> lowest_physical_corner_lower_zeta{
      {-1.0, 1.0, -1.0}};
  const std::array<double, 3> lowest_physical_corner_lower_eta{
      {-1.0, -1.0, 1.0}};
  const std::array<double, 3> lowest_physical_corner_lower_xi{
      {-1.0, 1.0, -1.0}};

  // Test that this map's logical axes point along +X, +Y, +Z:
  CHECK(map_upper_zeta(along_xi)[0] == approx(1.0));
  CHECK(map_upper_zeta(along_eta)[1] == approx(1.0));
  CHECK(map_upper_zeta(along_zeta)[2] == approx(2.0));
  CHECK_ITERABLE_APPROX(map_upper_zeta(lowest_corner),
                        lowest_physical_corner_upper_zeta);

  // Test that this map's logical axes point along +Z, +X, +Y:
  CHECK(map_upper_eta(along_xi)[2] == approx(1.0));
  CHECK(map_upper_eta(along_eta)[0] == approx(1.0));
  CHECK(map_upper_eta(along_zeta)[1] == approx(2.0));
  CHECK_ITERABLE_APPROX(map_upper_eta(lowest_corner),
                        lowest_physical_corner_upper_eta);

  // Test that this map's logical axes point along +Y, +Z, +X:
  CHECK(map_upper_xi(along_xi)[1] == approx(1.0));
  CHECK(map_upper_xi(along_eta)[2] == approx(1.0));
  CHECK(map_upper_xi(along_zeta)[0] == approx(2.0));
  CHECK_ITERABLE_APPROX(map_upper_xi(lowest_corner),
                        lowest_physical_corner_upper_xi);

  // Test that this map's logical axes point along +X, -Y, -Z:
  CHECK(map_lower_zeta(along_xi)[0] == approx(1.0));
  CHECK(map_lower_zeta(along_eta)[1] == approx(-1.0));
  CHECK(map_lower_zeta(along_zeta)[2] == approx(-2.0));
  CHECK_ITERABLE_APPROX(map_lower_zeta(lowest_corner),
                        lowest_physical_corner_lower_zeta);

  // Test that this map's logical axes point along -Z, +X, -Y:
  CHECK(map_lower_eta(along_xi)[2] == approx(-1.0));
  CHECK(map_lower_eta(along_eta)[0] == approx(1.0));
  CHECK(map_lower_eta(along_zeta)[1] == approx(-2.0));
  CHECK_ITERABLE_APPROX(map_lower_eta(lowest_corner),
                        lowest_physical_corner_lower_eta);

  // Test that this map's logical axes point along -Y, +Z, -X:
  CHECK(map_lower_xi(along_xi)[1] == approx(-1.0));
  CHECK(map_lower_xi(along_eta)[2] == approx(1.0));
  CHECK(map_lower_xi(along_zeta)[0] == approx(-2.0));
  CHECK_ITERABLE_APPROX(map_lower_xi(lowest_corner),
                        lowest_physical_corner_lower_xi);
}

void test_wedge3d_random_radii(const bool with_equiangular_map) {
  // Set up random number generator:
  const auto seed = std::random_device{}();
  std::mt19937 gen(seed);
  INFO("seed = " << seed);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(4, 7);

  // Check that points on the corners of the reference cube map to the correct
  // corners of the wedge.
  const std::array<double, 3> inner_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> outer_corner{{1.0, 1.0, 1.0}};
  const double random_inner_radius_lower_xi = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_lower_xi);
  const double random_inner_radius_lower_eta = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_lower_eta);
  const double random_inner_radius_lower_zeta = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_lower_zeta);
  const double random_inner_radius_upper_xi = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_upper_xi);
  const double random_inner_radius_upper_eta = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_upper_eta);
  const double random_inner_radius_upper_zeta = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_upper_zeta);

  const double random_outer_radius_lower_xi = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_lower_xi);
  const double random_outer_radius_lower_eta = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_lower_eta);
  const double random_outer_radius_lower_zeta = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_lower_zeta);
  const double random_outer_radius_upper_xi = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_upper_xi);
  const double random_outer_radius_upper_eta = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_upper_eta);
  const double random_outer_radius_upper_zeta = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_upper_zeta);

  const auto wedge_directions = all_wedge_directions();
  const CoordinateMaps::Wedge3D map_lower_xi(
      random_inner_radius_lower_xi, random_outer_radius_lower_xi,
      wedge_directions[5], 0.0, 1.0, with_equiangular_map);
  const CoordinateMaps::Wedge3D map_lower_eta(
      random_inner_radius_lower_eta, random_outer_radius_lower_eta,
      wedge_directions[3], 0.0, 1.0, with_equiangular_map);
  const CoordinateMaps::Wedge3D map_lower_zeta(
      random_inner_radius_lower_zeta, random_outer_radius_lower_zeta,
      wedge_directions[1], 0.0, 1.0, with_equiangular_map);
  const CoordinateMaps::Wedge3D map_upper_xi(
      random_inner_radius_upper_xi, random_outer_radius_upper_xi,
      wedge_directions[4], 0.0, 1.0, with_equiangular_map);
  const CoordinateMaps::Wedge3D map_upper_eta(
      random_inner_radius_upper_eta, random_outer_radius_upper_eta,
      wedge_directions[2], 0.0, 1.0, with_equiangular_map);
  const CoordinateMaps::Wedge3D map_upper_zeta(
      random_inner_radius_upper_zeta, random_outer_radius_upper_zeta,
      wedge_directions[0], 0.0, 1.0, with_equiangular_map);
  CHECK(map_lower_xi(outer_corner)[0] ==
        approx(-random_outer_radius_lower_xi / sqrt(3.0)));
  CHECK(map_lower_eta(outer_corner)[1] ==
        approx(-random_outer_radius_lower_eta / sqrt(3.0)));
  CHECK(map_lower_zeta(outer_corner)[2] ==
        approx(-random_outer_radius_lower_zeta / sqrt(3.0)));
  CHECK(map_upper_xi(inner_corner)[0] ==
        approx(random_inner_radius_upper_xi / sqrt(3.0)));
  CHECK(map_upper_eta(inner_corner)[1] ==
        approx(random_inner_radius_upper_eta / sqrt(3.0)));
  CHECK(map_upper_zeta(inner_corner)[2] ==
        approx(random_inner_radius_upper_zeta / sqrt(3.0)));

  // Check that random points on the edges of the reference cube map to the
  // correct edges of the wedge.
  const std::array<double, 3> random_outer_face{
      {real_dis(gen), real_dis(gen), 1.0}};
  const std::array<double, 3> random_inner_face{
      {real_dis(gen), real_dis(gen), -1.0}};
  CAPTURE_PRECISE(random_outer_face);
  CAPTURE_PRECISE(random_inner_face);

  CHECK(magnitude(map_lower_xi(random_outer_face)) ==
        approx(random_outer_radius_lower_xi));
  CHECK(map_lower_xi(random_inner_face)[0] ==
        approx(-random_inner_radius_lower_xi / sqrt(3.0)));
  CHECK(magnitude(map_lower_eta(random_outer_face)) ==
        approx(random_outer_radius_lower_eta));
  CHECK(map_lower_eta(random_inner_face)[1] ==
        approx(-random_inner_radius_lower_eta / sqrt(3.0)));
  CHECK(magnitude(map_upper_xi(random_outer_face)) ==
        approx(random_outer_radius_upper_xi));
  CHECK(map_upper_xi(random_inner_face)[0] ==
        approx(random_inner_radius_upper_xi / sqrt(3.0)));
  CHECK(magnitude(map_upper_eta(random_outer_face)) ==
        approx(random_outer_radius_upper_eta));
  CHECK(map_upper_eta(random_inner_face)[1] ==
        approx(random_inner_radius_upper_eta / sqrt(3.0)));
  CHECK(magnitude(map_lower_zeta(random_outer_face)) ==
        approx(random_outer_radius_lower_zeta));
  CHECK(magnitude(map_upper_zeta(random_outer_face)) ==
        approx(random_outer_radius_upper_zeta));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Map.Equiangular",
                  "[Domain][Unit]") {
  test_wedge3d_all_directions(true);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Map.Equidistant",
                  "[Domain][Unit]") {
  test_wedge3d_all_directions(false);
}
SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Alignment.Equiangular",
                  "[Domain][Unit]") {
  test_wedge3d_alignment(true);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Alignment.Equidistant",
                  "[Domain][Unit]") {
  test_wedge3d_alignment(false);
}
SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.RandomRadii.Equiangular",
                  "[Domain][Unit]") {
  test_wedge3d_random_radii(true);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.RandomRadii.Equidistant",
                  "[Domain][Unit]") {
  test_wedge3d_random_radii(false);
}

// [[OutputRegex, The radius of the inner surface must be greater than zero.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.RadiusInner",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge3d =
      CoordinateMaps::Wedge3D(-0.2, 4.0, OrientationMap<3>{}, 0.0, 1.0, true);
  static_cast<void>(failed_wedge3d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Sphericity of the inner surface must be between 0 and 1]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.SphericityInner",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge3d =
      CoordinateMaps::Wedge3D(0.2, 4.0, OrientationMap<3>{}, -0.2, 1.0, true);
  static_cast<void>(failed_wedge3d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Sphericity of the outer surface must be between 0 and 1]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.SphericityOuter",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge3d =
      CoordinateMaps::Wedge3D(0.2, 4.0, OrientationMap<3>{}, 0.0, -0.2, true);
  static_cast<void>(failed_wedge3d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The radius of the outer surface must be greater than the
// radius of the inner surface.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.RadiusOuter",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge3d =
      CoordinateMaps::Wedge3D(4.2, 4.0, OrientationMap<3>{}, 0.0, 1.0, true);
  static_cast<void>(failed_wedge3d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The arguments passed into the constructor for Wedge3D result
// in an object where the outer surface is pierced by the inner surface.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.PiercedSurface",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge3d =
      CoordinateMaps::Wedge3D(3.0, 4.0, OrientationMap<3>{}, 1.0, 0.0, true);
  static_cast<void>(failed_wedge3d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
