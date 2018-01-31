// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.ProductOf2Maps",
                  "[Domain][Unit]") {
  using affine_map = CoordinateMaps::Affine;
  using affine_map_2d = CoordinateMaps::ProductOf2Maps<affine_map, affine_map>;

  const double xA = -1.0;
  const double xB = 1.0;
  const double xa = -2.0;
  const double xb = 2.0;
  affine_map affine_map_x(xA, xB, xa, xb);

  const double yA = -2.0;
  const double yB = 3.0;
  const double ya = 5.0;
  const double yb = -2.0;

  affine_map affine_map_y(yA, yB, ya, yb);

  affine_map_2d affine_map_xy(affine_map_x, affine_map_y);

  const double xi = 0.5 * (xA + xB);
  const double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);
  const double eta = 0.5 * (yA + yB);
  const double y = yb * (eta - yA) / (yB - yA) + ya * (yB - eta) / (yB - yA);

  const std::array<double, 2> point_A{{xA, yA}};
  const std::array<double, 2> point_B{{xB, yB}};
  const std::array<double, 2> point_xi{{xi, eta}};
  const std::array<double, 2> point_a{{xa, ya}};
  const std::array<double, 2> point_b{{xb, yb}};
  const std::array<double, 2> point_x{{x, y}};

  CHECK(affine_map_xy(point_A) == point_a);
  CHECK(affine_map_xy(point_B) == point_b);

  CHECK(affine_map_xy.inverse(point_a) == point_A);
  CHECK(affine_map_xy.inverse(point_b) == point_B);

  const double inv_jacobian_00 = (xB - xA) / (xb - xa);
  const double inv_jacobian_11 = (yB - yA) / (yb - ya);
  const auto inv_jac_A = affine_map_xy.inv_jacobian(point_A);
  const auto inv_jac_B = affine_map_xy.inv_jacobian(point_B);
  const auto inv_jac_xi = affine_map_xy.inv_jacobian(point_xi);

  CHECK(inv_jac_A.get(0, 0) == inv_jacobian_00);
  CHECK(inv_jac_B.get(0, 0) == inv_jacobian_00);
  CHECK(inv_jac_xi.get(0, 0) == inv_jacobian_00);

  CHECK(inv_jac_A.get(0, 1) == 0.0);
  CHECK(inv_jac_B.get(0, 1) == 0.0);
  CHECK(inv_jac_xi.get(0, 1) == 0.0);

  CHECK(inv_jac_A.get(1, 0) == 0.0);
  CHECK(inv_jac_B.get(1, 0) == 0.0);
  CHECK(inv_jac_xi.get(1, 0) == 0.0);

  CHECK(inv_jac_A.get(1, 1) == inv_jacobian_11);
  CHECK(inv_jac_B.get(1, 1) == inv_jacobian_11);
  CHECK(inv_jac_xi.get(1, 1) == inv_jacobian_11);

  const double jacobian_00 = (xb - xa) / (xB - xA);
  const double jacobian_11 = (yb - ya) / (yB - yA);
  const auto jac_A = affine_map_xy.jacobian(point_A);
  const auto jac_B = affine_map_xy.jacobian(point_B);
  const auto jac_xi = affine_map_xy.jacobian(point_xi);

  CHECK(jac_A.get(0, 0) == jacobian_00);
  CHECK(jac_B.get(0, 0) == jacobian_00);
  CHECK(jac_xi.get(0, 0) == jacobian_00);

  CHECK(jac_A.get(0, 1) == 0.0);
  CHECK(jac_B.get(0, 1) == 0.0);
  CHECK(jac_xi.get(0, 1) == 0.0);

  CHECK(jac_A.get(1, 0) == 0.0);
  CHECK(jac_B.get(1, 0) == 0.0);
  CHECK(jac_xi.get(1, 0) == 0.0);

  CHECK(jac_A.get(1, 1) == jacobian_11);
  CHECK(jac_B.get(1, 1) == jacobian_11);
  CHECK(jac_xi.get(1, 1) == jacobian_11);

  // Check Jacobians for DataVectors
  const Index<2> extents(8);
  const auto tensor_logical_coords = logical_coordinates(extents);
  const std::array<DataVector, 2> logical_coords{
      {tensor_logical_coords.get(0), tensor_logical_coords.get(1)}};
  const auto volume_inv_jac = affine_map_xy.inv_jacobian(logical_coords);
  const auto volume_jac = affine_map_xy.jacobian(logical_coords);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      if (i == j) {
        CHECK(volume_inv_jac.get(i, j) ==
              DataVector(logical_coords[0].size(),
                         i == 0 ? inv_jacobian_00 : inv_jacobian_11));
        CHECK(volume_jac.get(i, j) ==
              DataVector(logical_coords[0].size(),
                         i == 0 ? jacobian_00 : jacobian_11));
      } else {
        CHECK(volume_inv_jac.get(i, j) ==
              DataVector(logical_coords[0].size(), 0.0));
        CHECK(volume_jac.get(i, j) ==
              DataVector(logical_coords[0].size(), 0.0));
      }
    }
  }

  CHECK_FALSE(affine_map_xy != affine_map_xy);
  test_serialization(affine_map_xy);

  test_coordinate_map_argument_types(affine_map_xy, point_xi);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.ProductOf3Maps",
                  "[Domain][Unit]") {
  using affine_map = CoordinateMaps::Affine;
  using affine_map_3d =
      CoordinateMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;

  const double xA = -1.0;
  const double xB = 1.0;
  const double xa = -2.0;
  const double xb = 2.0;
  affine_map affine_map_x(xA, xB, xa, xb);

  const double yA = -2.0;
  const double yB = 3.0;
  const double ya = 5.0;
  const double yb = -2.0;

  affine_map affine_map_y(yA, yB, ya, yb);

  const double zA = 4.0;
  const double zB = 8.0;
  const double za = -15.0;
  const double zb = 12.0;

  affine_map affine_map_z(zA, zB, za, zb);

  affine_map_3d affine_map_xyz(affine_map_x, affine_map_y, affine_map_z);

  const double xi = 0.5 * (xA + xB);
  const double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);
  const double eta = 0.5 * (yA + yB);
  const double y = yb * (eta - yA) / (yB - yA) + ya * (yB - eta) / (yB - yA);
  const double zeta = 0.5 * (zA + zB);
  const double z = zb * (eta - zA) / (zB - zA) + za * (zB - eta) / (zB - zA);

  const std::array<double, 3> point_A{{xA, yA, zA}};
  const std::array<double, 3> point_B{{xB, yB, zB}};
  const std::array<double, 3> point_xi{{xi, eta, zeta}};
  const std::array<double, 3> point_a{{xa, ya, za}};
  const std::array<double, 3> point_b{{xb, yb, zb}};
  const std::array<double, 3> point_x{{x, y, z}};

  CHECK(affine_map_xyz(point_A) == point_a);
  CHECK(affine_map_xyz(point_B) == point_b);

  CHECK(affine_map_xyz.inverse(point_a) == point_A);
  CHECK(affine_map_xyz.inverse(point_b) == point_B);

  const double inv_jacobian_00 = (xB - xA) / (xb - xa);
  const double inv_jacobian_11 = (yB - yA) / (yb - ya);
  const double inv_jacobian_22 = (zB - zA) / (zb - za);
  const auto inv_jac_A = affine_map_xyz.inv_jacobian(point_A);
  const auto inv_jac_B = affine_map_xyz.inv_jacobian(point_B);
  const auto inv_jac_xi = affine_map_xyz.inv_jacobian(point_xi);

  CHECK(inv_jac_A.get(0, 0) == inv_jacobian_00);
  CHECK(inv_jac_B.get(0, 0) == inv_jacobian_00);
  CHECK(inv_jac_xi.get(0, 0) == inv_jacobian_00);

  CHECK(inv_jac_A.get(0, 1) == 0.0);
  CHECK(inv_jac_B.get(0, 1) == 0.0);
  CHECK(inv_jac_xi.get(0, 1) == 0.0);

  CHECK(inv_jac_A.get(0, 2) == 0.0);
  CHECK(inv_jac_B.get(0, 2) == 0.0);
  CHECK(inv_jac_xi.get(0, 2) == 0.0);

  CHECK(inv_jac_A.get(1, 0) == 0.0);
  CHECK(inv_jac_B.get(1, 0) == 0.0);
  CHECK(inv_jac_xi.get(1, 0) == 0.0);

  CHECK(inv_jac_A.get(1, 1) == inv_jacobian_11);
  CHECK(inv_jac_B.get(1, 1) == inv_jacobian_11);
  CHECK(inv_jac_xi.get(1, 1) == inv_jacobian_11);

  CHECK(inv_jac_A.get(1, 2) == 0.0);
  CHECK(inv_jac_B.get(1, 2) == 0.0);
  CHECK(inv_jac_xi.get(1, 2) == 0.0);

  CHECK(inv_jac_A.get(2, 0) == 0.0);
  CHECK(inv_jac_B.get(2, 0) == 0.0);
  CHECK(inv_jac_xi.get(2, 0) == 0.0);

  CHECK(inv_jac_A.get(2, 1) == 0.0);
  CHECK(inv_jac_B.get(2, 1) == 0.0);
  CHECK(inv_jac_xi.get(2, 1) == 0.0);

  CHECK(inv_jac_A.get(2, 2) == inv_jacobian_22);
  CHECK(inv_jac_B.get(2, 2) == inv_jacobian_22);
  CHECK(inv_jac_xi.get(2, 2) == inv_jacobian_22);

  const double jacobian_00 = (xb - xa) / (xB - xA);
  const double jacobian_11 = (yb - ya) / (yB - yA);
  const double jacobian_22 = (zb - za) / (zB - zA);
  const auto jac_A = affine_map_xyz.jacobian(point_A);
  const auto jac_B = affine_map_xyz.jacobian(point_B);
  const auto jac_xi = affine_map_xyz.jacobian(point_xi);

  CHECK(jac_A.get(0, 0) == jacobian_00);
  CHECK(jac_B.get(0, 0) == jacobian_00);
  CHECK(jac_xi.get(0, 0) == jacobian_00);

  CHECK(jac_A.get(0, 1) == 0.0);
  CHECK(jac_B.get(0, 1) == 0.0);
  CHECK(jac_xi.get(0, 1) == 0.0);

  CHECK(jac_A.get(0, 2) == 0.0);
  CHECK(jac_B.get(0, 2) == 0.0);
  CHECK(jac_xi.get(0, 2) == 0.0);

  CHECK(jac_A.get(1, 0) == 0.0);
  CHECK(jac_B.get(1, 0) == 0.0);
  CHECK(jac_xi.get(1, 0) == 0.0);

  CHECK(jac_A.get(1, 1) == jacobian_11);
  CHECK(jac_B.get(1, 1) == jacobian_11);
  CHECK(jac_xi.get(1, 1) == jacobian_11);

  CHECK(jac_A.get(1, 2) == 0.0);
  CHECK(jac_B.get(1, 2) == 0.0);
  CHECK(jac_xi.get(1, 2) == 0.0);

  CHECK(jac_A.get(2, 0) == 0.0);
  CHECK(jac_B.get(2, 0) == 0.0);
  CHECK(jac_xi.get(2, 0) == 0.0);

  CHECK(jac_A.get(2, 1) == 0.0);
  CHECK(jac_B.get(2, 1) == 0.0);
  CHECK(jac_xi.get(2, 1) == 0.0);

  CHECK(jac_A.get(2, 2) == jacobian_22);
  CHECK(jac_B.get(2, 2) == jacobian_22);
  CHECK(jac_xi.get(2, 2) == jacobian_22);

  // Check Jacobians for DataVectors
  const Index<3> extents(8);
  const auto tensor_logical_coords = logical_coordinates(extents);
  const std::array<DataVector, 3> logical_coords{
      {tensor_logical_coords.get(0), tensor_logical_coords.get(1),
       tensor_logical_coords.get(2)}};
  const auto volume_inv_jac = affine_map_xyz.inv_jacobian(logical_coords);
  const auto volume_jac = affine_map_xyz.jacobian(logical_coords);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        CHECK(volume_inv_jac.get(i, j) ==
              DataVector(logical_coords[0].size(),
                         i == 0 ? inv_jacobian_00
                                : i == 1 ? inv_jacobian_11 : inv_jacobian_22));
        CHECK(volume_jac.get(i, j) ==
              DataVector(
                  logical_coords[0].size(),
                  i == 0 ? jacobian_00 : i == 1 ? jacobian_11 : jacobian_22));
      } else {
        CHECK(volume_inv_jac.get(i, j) ==
              DataVector(logical_coords[0].size(), 0.0));
        CHECK(volume_jac.get(i, j) ==
              DataVector(logical_coords[0].size(), 0.0));
      }
    }
  }

  CHECK_FALSE(affine_map_xyz != affine_map_xyz);
  test_serialization(affine_map_xyz);

  test_coordinate_map_argument_types(affine_map_xyz, point_xi);
}
