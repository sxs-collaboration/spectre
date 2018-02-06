// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <catch.hpp>
#include <random>

#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.BulgedCube.Identity",
                  "[Domain][Unit]") {
  const CoordinateMaps::BulgedCube map(sqrt(3.0), 0, false);
  const std::array<double, 3> lower_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> upper_corner{{1.0, 1.0, 1.0}};
  const std::array<double, 3> test_point1{{-1.0, 0.25, 0.0}};
  const std::array<double, 3> test_point2{{1.0, 1.0, -0.5}};
  const std::array<double, 3> test_point3{{0.7, -0.2, 0.4}};

  CHECK_ITERABLE_APPROX(map(lower_corner), lower_corner);
  CHECK_ITERABLE_APPROX(map(upper_corner), upper_corner);
  CHECK_ITERABLE_APPROX(map(test_point1), test_point1);
  CHECK_ITERABLE_APPROX(map(test_point2), test_point2);
  CHECK_ITERABLE_APPROX(map(test_point3), test_point3);

  test_jacobian(map, test_point1);
  test_jacobian(map, test_point2);
  test_jacobian(map, test_point3);

  test_inv_jacobian(map, test_point1);
  test_inv_jacobian(map, test_point2);
  test_inv_jacobian(map, test_point3);

  test_coordinate_map_implementation<CoordinateMaps::BulgedCube>(map);

  CHECK(serialize_and_deserialize(map) == map);
  CHECK_FALSE(serialize_and_deserialize(map) != map);

  test_coordinate_map_argument_types<true>(map, test_point1);
}

void test_bulged_cube(bool with_equiangular_map) {
  const std::array<double, 3> lower_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> upper_corner{{1.0, 1.0, 1.0}};
  const CoordinateMaps::BulgedCube map(2.0 * sqrt(3.0), 0.5,
                                       with_equiangular_map);

  CHECK(map(lower_corner)[0] == approx(-2.0));
  CHECK(map(lower_corner)[1] == approx(-2.0));
  CHECK(map(lower_corner)[2] == approx(-2.0));
  CHECK(map(upper_corner)[0] == approx(2.0));
  CHECK(map(upper_corner)[1] == approx(2.0));
  CHECK(map(upper_corner)[2] == approx(2.0));

  const std::array<double, 3> test_point1{{-1.0, 0.25, 0.0}};
  const std::array<double, 3> test_point2{{1.0, 1.0, -0.5}};
  const std::array<double, 3> test_point3{{0.7, -0.2, 0.4}};
  const std::array<double, 3> test_point4{{0.0, 0.0, 0.0}};
  const std::array<DataVector, 3> test_points{
      {DataVector{-1.0, 1.0, 0.7, 0.0}, DataVector{0.25, 1.0, -0.2, 0.0},
       DataVector{0.0, -0.5, 0.4, 0.0}}};
  const DataVector& collocation_pts = Basis::lgl::collocation_points(7);
  const std::array<DataVector, 3> test_points2{
      {collocation_pts, collocation_pts, collocation_pts}};

  test_jacobian(map, test_point1);
  test_jacobian(map, test_point2);
  test_jacobian(map, test_point3);
  test_jacobian(map, test_point4);

  test_inv_jacobian(map, test_point1);
  test_inv_jacobian(map, test_point2);
  test_inv_jacobian(map, test_point3);
  test_inv_jacobian(map, test_point4);

  test_coordinate_map_implementation<CoordinateMaps::BulgedCube>(map);

  CHECK(serialize_and_deserialize(map) == map);
  CHECK_FALSE(serialize_and_deserialize(map) != map);

  test_coordinate_map_argument_types<true>(map, test_point1);

  test_inverse_map(map, test_point1);
  test_inverse_map(map, test_point2);
  test_inverse_map(map, test_point3);
  test_inverse_map(map, test_point4);
  test_inverse_map(map, test_points);
  test_inverse_map(map, test_points2);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.BulgedCube.Bulged.Equiangular",
                  "[Domain][Unit]") {
  test_bulged_cube(true);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.BulgedCube.Bulged.Equidistant",
                  "[Domain][Unit]") {
  test_bulged_cube(false);
}
