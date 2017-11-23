// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <catch.hpp>
#include <random>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Sphere.Equiangular",
                  "[Domain][Unit]") {
  const std::array<double, 3> lower_corner{{1.0, 1.0, -1.0}};
  const std::array<double, 3> upper_corner{{1.0, 1.0, 1.0}};
  const std::array<double, 3> test_point1{{-0.1, 0.3, 0.1}};
  const std::array<double, 3> test_point2{{0.5, 0.7, -0.5}};
  const std::array<double, 3> test_point3{{0.9, -1.0, 0.4}};

  for (const auto& direction : Direction<3>::all_directions()) {
    const CoordinateMaps::Wedge3D map(sqrt(3.0), 2.0 * sqrt(3.0), direction, 1,
                                      true);
    test_jacobian(map, test_point1);
    test_jacobian(map, test_point2);
    test_jacobian(map, test_point3);

    test_inv_jacobian(map, test_point1);
    test_inv_jacobian(map, test_point2);
    test_inv_jacobian(map, test_point3);

    test_inverse_map(map, test_point1);
    test_inverse_map(map, test_point2);
    test_inverse_map(map, test_point3);
  }

  const CoordinateMaps::Wedge3D map_upper_zeta(
      sqrt(3.0), 2.0 * sqrt(3.0), Direction<3>::upper_zeta(), 1, true);

  CHECK(map_upper_zeta(lower_corner)[0] == approx(1.0));
  CHECK(map_upper_zeta(lower_corner)[1] == approx(1.0));
  CHECK(map_upper_zeta(lower_corner)[2] == approx(1.0));
  CHECK(map_upper_zeta(upper_corner)[0] == approx(2.0));
  CHECK(map_upper_zeta(upper_corner)[1] == approx(2.0));
  CHECK(map_upper_zeta(upper_corner)[2] == approx(2.0));

  test_coordinate_map_implementation<CoordinateMaps::Wedge3D>(map_upper_zeta);

  test_serialization(map_upper_zeta);
  CHECK_FALSE(map_upper_zeta != map_upper_zeta);

  test_coordinate_map_argument_types(map_upper_zeta, test_point1);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Sphere.Equidistant",
                  "[Domain][Unit]") {
  const std::array<double, 3> lower_corner{{1.0, 1.0, -1.0}};
  const std::array<double, 3> upper_corner{{1.0, 1.0, 1.0}};
  const std::array<double, 3> test_point1{{-0.1, 0.3, 0.1}};
  const std::array<double, 3> test_point2{{0.5, 0.7, -0.5}};
  const std::array<double, 3> test_point3{{0.9, -1.0, 0.4}};

  for (const auto& direction : Direction<3>::all_directions()) {
    const CoordinateMaps::Wedge3D map(sqrt(3.0), 2.0 * sqrt(3.0), direction, 1,
                                      false);
    test_jacobian(map, test_point1);
    test_jacobian(map, test_point2);
    test_jacobian(map, test_point3);

    test_inv_jacobian(map, test_point1);
    test_inv_jacobian(map, test_point2);
    test_inv_jacobian(map, test_point3);

    test_inverse_map(map, test_point1);
    test_inverse_map(map, test_point2);
    test_inverse_map(map, test_point3);
  }

  const CoordinateMaps::Wedge3D map_upper_zeta(
      sqrt(3.0), 2.0 * sqrt(3.0), Direction<3>::upper_zeta(), 1, false);

  CHECK(map_upper_zeta(lower_corner)[0] == approx(1.0));
  CHECK(map_upper_zeta(lower_corner)[1] == approx(1.0));
  CHECK(map_upper_zeta(lower_corner)[2] == approx(1.0));
  CHECK(map_upper_zeta(upper_corner)[0] == approx(2.0));
  CHECK(map_upper_zeta(upper_corner)[1] == approx(2.0));
  CHECK(map_upper_zeta(upper_corner)[2] == approx(2.0));

  test_coordinate_map_implementation<CoordinateMaps::Wedge3D>(map_upper_zeta);

  test_serialization(map_upper_zeta);
  CHECK_FALSE(map_upper_zeta != map_upper_zeta);

  test_coordinate_map_argument_types(map_upper_zeta, test_point1);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Alignment.Equiangular",
                  "[Domain][Unit]") {
  // Test that the logical axes point along the expected directions in
  // physical space

  const double inner_r = sqrt(3.0);
  const double outer_r = 2.0 * sqrt(3.0);

  const CoordinateMaps::Wedge3D map_upper_zeta(
      inner_r, outer_r, Direction<3>::upper_zeta(), 0, true);  // Upper Z wedge
  const CoordinateMaps::Wedge3D map_upper_eta(
      inner_r, outer_r, Direction<3>::upper_eta(), 0, true);  // Upper Y wedge
  const CoordinateMaps::Wedge3D map_upper_xi(
      inner_r, outer_r, Direction<3>::upper_xi(), 0, true);  // Upper X Wedge
  const CoordinateMaps::Wedge3D map_lower_zeta(
      inner_r, outer_r, Direction<3>::lower_zeta(), 0, true);  // Lower Z wedge
  const CoordinateMaps::Wedge3D map_lower_eta(
      inner_r, outer_r, Direction<3>::lower_eta(), 0, true);  // Lower Y wedge
  const CoordinateMaps::Wedge3D map_lower_xi(
      inner_r, outer_r, Direction<3>::lower_xi(), 0, true);  // Lower X wedge
  const std::array<double, 3> lowest_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> along_xi{{1.0, -1.0, -1.0}};
  const std::array<double, 3> along_eta{{-1.0, 1.0, -1.0}};
  const std::array<double, 3> along_zeta{{-1.0, -1.0, 1.0}};

  // Test that this map's logical axes point along +X, +Y, +Z:
  CHECK(map_upper_zeta(along_xi)[0] == approx(1.0));
  CHECK(map_upper_zeta(lowest_corner)[0] == approx(-1.0));
  CHECK(map_upper_zeta(along_eta)[1] == approx(1.0));
  CHECK(map_upper_zeta(lowest_corner)[1] == approx(-1.0));
  CHECK(map_upper_zeta(along_zeta)[2] == approx(2.0));
  CHECK(map_upper_zeta(lowest_corner)[2] == approx(1.0));

  // Test that this map's logical axes point along +Z, +X, +Y:
  CHECK(map_upper_eta(along_xi)[2] == approx(1.0));
  CHECK(map_upper_eta(lowest_corner)[2] == approx(-1.0));
  CHECK(map_upper_eta(along_eta)[0] == approx(1.0));
  CHECK(map_upper_eta(lowest_corner)[0] == approx(-1.0));
  CHECK(map_upper_eta(along_zeta)[1] == approx(2.0));
  CHECK(map_upper_eta(lowest_corner)[1] == approx(1.0));

  // Test that this map's logical axes point along +Y, +Z, +X:
  CHECK(map_upper_xi(along_xi)[1] == approx(1.0));
  CHECK(map_upper_xi(lowest_corner)[1] == approx(-1.0));
  CHECK(map_upper_xi(along_eta)[2] == approx(1.0));
  CHECK(map_upper_xi(lowest_corner)[2] == approx(-1.0));
  CHECK(map_upper_xi(along_zeta)[0] == approx(2.0));
  CHECK(map_upper_xi(lowest_corner)[0] == approx(1.0));

  // Test that this map's logical axes point along +X, -Y, -Z:
  CHECK(map_lower_zeta(along_xi)[0] == approx(1.0));
  CHECK(map_lower_zeta(lowest_corner)[0] == approx(-1.0));
  CHECK(map_lower_zeta(along_eta)[1] == approx(-1.0));
  CHECK(map_lower_zeta(lowest_corner)[1] == approx(1.0));
  CHECK(map_lower_zeta(along_zeta)[2] == approx(-2.0));
  CHECK(map_lower_zeta(lowest_corner)[2] == approx(-1.0));

  // Test that this map's logical axes point along -Z, +X, -Y:
  CHECK(map_lower_eta(along_xi)[2] == approx(-1.0));
  CHECK(map_lower_eta(lowest_corner)[2] == approx(1.0));
  CHECK(map_lower_eta(along_eta)[0] == approx(1.0));
  CHECK(map_lower_eta(lowest_corner)[0] == approx(-1.0));
  CHECK(map_lower_eta(along_zeta)[1] == approx(-2.0));
  CHECK(map_lower_eta(lowest_corner)[1] == approx(-1.0));

  // Test that this map's logical axes point along -Y, +Z, -X:
  CHECK(map_lower_xi(along_xi)[1] == approx(-1.0));
  CHECK(map_lower_xi(lowest_corner)[1] == approx(1.0));
  CHECK(map_lower_xi(along_eta)[2] == approx(1.0));
  CHECK(map_lower_xi(lowest_corner)[2] == approx(-1.0));
  CHECK(map_lower_xi(along_zeta)[0] == approx(-2.0));
  CHECK(map_lower_xi(lowest_corner)[0] == approx(-1.0));
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Alignment.Equidistant",
                  "[Domain][Unit]") {
  // Test that the logical axes point along the expected directions in
  // physical space

  const double inner_r = sqrt(3.0);
  const double outer_r = 2.0 * sqrt(3.0);

  const CoordinateMaps::Wedge3D map_upper_zeta(
      inner_r, outer_r, Direction<3>::upper_zeta(), 0, true);  // Upper Z wedge
  const CoordinateMaps::Wedge3D map_upper_eta(
      inner_r, outer_r, Direction<3>::upper_eta(), 0, true);  // Upper Y wedge
  const CoordinateMaps::Wedge3D map_upper_xi(
      inner_r, outer_r, Direction<3>::upper_xi(), 0, true);  // Upper X Wedge
  const CoordinateMaps::Wedge3D map_lower_zeta(
      inner_r, outer_r, Direction<3>::lower_zeta(), 0, true);  // Lower Z wedge
  const CoordinateMaps::Wedge3D map_lower_eta(
      inner_r, outer_r, Direction<3>::lower_eta(), 0, true);  // Lower Y wedge
  const CoordinateMaps::Wedge3D map_lower_xi(
      inner_r, outer_r, Direction<3>::lower_xi(), 0, true);  // Lower X wedge
  const std::array<double, 3> lowest_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> along_xi{{1.0, -1.0, -1.0}};
  const std::array<double, 3> along_eta{{-1.0, 1.0, -1.0}};
  const std::array<double, 3> along_zeta{{-1.0, -1.0, 1.0}};

  // Test that this map's logical axes point along +X, +Y, +Z:
  CHECK(map_upper_zeta(along_xi)[0] == approx(1.0));
  CHECK(map_upper_zeta(lowest_corner)[0] == approx(-1.0));
  CHECK(map_upper_zeta(along_eta)[1] == approx(1.0));
  CHECK(map_upper_zeta(lowest_corner)[1] == approx(-1.0));
  CHECK(map_upper_zeta(along_zeta)[2] == approx(2.0));
  CHECK(map_upper_zeta(lowest_corner)[2] == approx(1.0));

  // Test that this map's logical axes point along +Z, +X, +Y:
  CHECK(map_upper_eta(along_xi)[2] == approx(1.0));
  CHECK(map_upper_eta(lowest_corner)[2] == approx(-1.0));
  CHECK(map_upper_eta(along_eta)[0] == approx(1.0));
  CHECK(map_upper_eta(lowest_corner)[0] == approx(-1.0));
  CHECK(map_upper_eta(along_zeta)[1] == approx(2.0));
  CHECK(map_upper_eta(lowest_corner)[1] == approx(1.0));

  // Test that this map's logical axes point along +Y, +Z, +X:
  CHECK(map_upper_xi(along_xi)[1] == approx(1.0));
  CHECK(map_upper_xi(lowest_corner)[1] == approx(-1.0));
  CHECK(map_upper_xi(along_eta)[2] == approx(1.0));
  CHECK(map_upper_xi(lowest_corner)[2] == approx(-1.0));
  CHECK(map_upper_xi(along_zeta)[0] == approx(2.0));
  CHECK(map_upper_xi(lowest_corner)[0] == approx(1.0));

  // Test that this map's logical axes point along +X, -Y, -Z:
  CHECK(map_lower_zeta(along_xi)[0] == approx(1.0));
  CHECK(map_lower_zeta(lowest_corner)[0] == approx(-1.0));
  CHECK(map_lower_zeta(along_eta)[1] == approx(-1.0));
  CHECK(map_lower_zeta(lowest_corner)[1] == approx(1.0));
  CHECK(map_lower_zeta(along_zeta)[2] == approx(-2.0));
  CHECK(map_lower_zeta(lowest_corner)[2] == approx(-1.0));

  // Test that this map's logical axes point along -Z, +X, -Y:
  CHECK(map_lower_eta(along_xi)[2] == approx(-1.0));
  CHECK(map_lower_eta(lowest_corner)[2] == approx(1.0));
  CHECK(map_lower_eta(along_eta)[0] == approx(1.0));
  CHECK(map_lower_eta(lowest_corner)[0] == approx(-1.0));
  CHECK(map_lower_eta(along_zeta)[1] == approx(-2.0));
  CHECK(map_lower_eta(lowest_corner)[1] == approx(-1.0));

  // Test that this map's logical axes point along -Y, +Z, -X:
  CHECK(map_lower_xi(along_xi)[1] == approx(-1.0));
  CHECK(map_lower_xi(lowest_corner)[1] == approx(1.0));
  CHECK(map_lower_xi(along_eta)[2] == approx(1.0));
  CHECK(map_lower_xi(lowest_corner)[2] == approx(-1.0));
  CHECK(map_lower_xi(along_zeta)[0] == approx(-2.0));
  CHECK(map_lower_xi(lowest_corner)[0] == approx(-1.0));
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.RandomRadii.Equiangular",
                  "[Domain][Unit]") {
  // Set up random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
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

  const CoordinateMaps::Wedge3D map_lower_xi(random_inner_radius_lower_xi,
                                             random_outer_radius_lower_xi,
                                             Direction<3>::lower_xi(), 0, true);
  const CoordinateMaps::Wedge3D map_lower_eta(
      random_inner_radius_lower_eta, random_outer_radius_lower_eta,
      Direction<3>::lower_eta(), 0, true);
  const CoordinateMaps::Wedge3D map_lower_zeta(
      random_inner_radius_lower_zeta, random_outer_radius_lower_zeta,
      Direction<3>::lower_zeta(), 0, true);
  const CoordinateMaps::Wedge3D map_upper_xi(random_inner_radius_upper_xi,
                                             random_outer_radius_upper_xi,
                                             Direction<3>::upper_xi(), 0, true);
  const CoordinateMaps::Wedge3D map_upper_eta(
      random_inner_radius_upper_eta, random_outer_radius_upper_eta,
      Direction<3>::upper_eta(), 0, true);
  const CoordinateMaps::Wedge3D map_upper_zeta(
      random_inner_radius_upper_zeta, random_outer_radius_upper_zeta,
      Direction<3>::upper_zeta(), 0, true);
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
  for (size_t i = 0; i < 3; ++i) {
    CAPTURE_PRECISE(gsl::at(random_outer_face, i));
    CAPTURE_PRECISE(gsl::at(random_inner_face, i));
  }

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

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.RandomRadii.Equidistant",
                  "[Domain][Unit]") {
  // Set up random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
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

  const CoordinateMaps::Wedge3D map_lower_xi(
      random_inner_radius_lower_xi, random_outer_radius_lower_xi,
      Direction<3>::lower_xi(), 0, false);
  const CoordinateMaps::Wedge3D map_lower_eta(
      random_inner_radius_lower_eta, random_outer_radius_lower_eta,
      Direction<3>::lower_eta(), 0, false);
  const CoordinateMaps::Wedge3D map_lower_zeta(
      random_inner_radius_lower_zeta, random_outer_radius_lower_zeta,
      Direction<3>::lower_zeta(), 0, false);
  const CoordinateMaps::Wedge3D map_upper_xi(
      random_inner_radius_upper_xi, random_outer_radius_upper_xi,
      Direction<3>::upper_xi(), 0, false);
  const CoordinateMaps::Wedge3D map_upper_eta(
      random_inner_radius_upper_eta, random_outer_radius_upper_eta,
      Direction<3>::upper_eta(), 0, false);
  const CoordinateMaps::Wedge3D map_upper_zeta(
      random_inner_radius_upper_zeta, random_outer_radius_upper_zeta,
      Direction<3>::upper_zeta(), 0, false);
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
  for (size_t i = 0; i < 3; ++i) {
    CAPTURE_PRECISE(gsl::at(random_outer_face, i));
    CAPTURE_PRECISE(gsl::at(random_inner_face, i));
  }

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

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Jacobian.Equiangular",
                  "[Domain][Unit]") {
  // Set up random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> unit_dis(0, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(4, 7);

  const double xi = real_dis(gen);
  CAPTURE_PRECISE(xi);
  const double eta = real_dis(gen);
  CAPTURE_PRECISE(eta);
  const double zeta = real_dis(gen);
  CAPTURE_PRECISE(zeta);
  const double inner_radius = inner_dis(gen);
  CAPTURE_PRECISE(inner_radius);
  const double outer_radius = outer_dis(gen);
  CAPTURE_PRECISE(outer_radius);
  const double sphericity = unit_dis(gen);
  CAPTURE_PRECISE(sphericity);

  for (const auto direction : Direction<3>::all_directions()) {
    const CoordinateMaps::Wedge3D map{inner_radius, outer_radius, direction,
                                      sphericity, true};
    const CoordinateMaps::Wedge3D map_spheres{inner_radius, outer_radius,
                                              direction, 1, true};

    const std::array<double, 3> test_point{{xi, eta, zeta}};
    test_jacobian(map, test_point);
    test_inv_jacobian(map, test_point);
    test_jacobian(map_spheres, test_point);
    test_inv_jacobian(map_spheres, test_point);
  }
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Jacobian.Equidistant",
                  "[Domain][Unit]") {
  // Set up random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> unit_dis(0, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(4, 7);

  const double xi = real_dis(gen);
  CAPTURE_PRECISE(xi);
  const double eta = real_dis(gen);
  CAPTURE_PRECISE(eta);
  const double zeta = real_dis(gen);
  CAPTURE_PRECISE(zeta);
  const double inner_radius = inner_dis(gen);
  CAPTURE_PRECISE(inner_radius);
  const double outer_radius = outer_dis(gen);
  CAPTURE_PRECISE(outer_radius);
  const double sphericity = unit_dis(gen);
  CAPTURE_PRECISE(sphericity);

  for (const auto direction : Direction<3>::all_directions()) {
    const CoordinateMaps::Wedge3D map{inner_radius, outer_radius, direction,
                                      sphericity, false};
    const CoordinateMaps::Wedge3D map_spheres{inner_radius, outer_radius,
                                              direction, 1, false};

    const std::array<double, 3> test_point{{xi, eta, zeta}};
    test_jacobian(map, test_point);
    test_inv_jacobian(map, test_point);
    test_jacobian(map_spheres, test_point);
    test_inv_jacobian(map_spheres, test_point);
  }
}
