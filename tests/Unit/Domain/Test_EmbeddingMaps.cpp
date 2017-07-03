// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/EmbeddingMaps/AffineMap.hpp"
#include "Domain/EmbeddingMaps/Identity.hpp"
#include "Domain/EmbeddingMaps/ProductMaps.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

TEST_CASE("Unit.Domain.EmbeddingMaps.ProductOf2Maps", "[Domain][Unit]") {
  double xA = -1.0;
  double xB = 1.0;
  double xa = -2.0;
  double xb = 2.0;
  EmbeddingMaps::AffineMap affine_map_x(xA, xB, xa, xb);

  double yA = -2.0;
  double yB = 3.0;
  double ya = 5.0;
  double yb = -2.0;

  EmbeddingMaps::AffineMap affine_map_y(yA, yB, ya, yb);

  EmbeddingMaps::ProductOf2Maps<1, 1> affine_map_xy(affine_map_x, affine_map_y);

  double xi = 0.5 * (xA + xB);
  double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);
  double eta = 0.5 * (yA + yB);
  double y = yb * (eta - yA) / (yB - yA) + ya * (yB - eta) / (yB - yA);

  Point<2, Frame::Logical> point_A{{xA, yA}};
  Point<2, Frame::Logical> point_B{{xB, yB}};
  Point<2, Frame::Logical> point_xi{{xi, eta}};
  Point<2, Frame::Grid> point_a{{xa, ya}};
  Point<2, Frame::Grid> point_b{{xb, yb}};
  Point<2, Frame::Grid> point_x{{x, y}};

  CHECK(affine_map_xy(point_A) == point_a);
  CHECK(affine_map_xy(point_B) == point_b);

  CHECK(affine_map_xy.inverse(point_a) == point_A);
  CHECK(affine_map_xy.inverse(point_b) == point_B);

  double invJ00 = (xB - xA) / (xb - xa);
  double invJ11 = (yB - yA) / (yb - ya);

  CHECK(affine_map_xy.inv_jacobian(point_A, 0, 0) == invJ00);
  CHECK(affine_map_xy.inv_jacobian(point_B, 0, 0) == invJ00);
  CHECK(affine_map_xy.inv_jacobian(point_xi, 0, 0) == invJ00);

  CHECK(affine_map_xy.inv_jacobian(point_A, 0, 1) == 0.0);
  CHECK(affine_map_xy.inv_jacobian(point_B, 0, 1) == 0.0);
  CHECK(affine_map_xy.inv_jacobian(point_xi, 0, 1) == 0.0);

  CHECK(affine_map_xy.inv_jacobian(point_A, 1, 0) == 0.0);
  CHECK(affine_map_xy.inv_jacobian(point_B, 1, 0) == 0.0);
  CHECK(affine_map_xy.inv_jacobian(point_xi, 1, 0) == 0.0);

  CHECK(affine_map_xy.inv_jacobian(point_A, 1, 1) == invJ11);
  CHECK(affine_map_xy.inv_jacobian(point_B, 1, 1) == invJ11);
  CHECK(affine_map_xy.inv_jacobian(point_xi, 1, 1) == invJ11);
}

TEST_CASE("Unit.Domain.EmbeddingMaps.ProductOf3Maps", "[Domain][Unit]") {
  double xA = -1.0;
  double xB = 1.0;
  double xa = -2.0;
  double xb = 2.0;
  EmbeddingMaps::AffineMap affine_map_x(xA, xB, xa, xb);

  double yA = -2.0;
  double yB = 3.0;
  double ya = 5.0;
  double yb = -2.0;

  EmbeddingMaps::AffineMap affine_map_y(yA, yB, ya, yb);

  double zA = 4.0;
  double zB = 8.0;
  double za = -15.0;
  double zb = 12.0;

  EmbeddingMaps::AffineMap affine_map_z(zA, zB, za, zb);

  EmbeddingMaps::ProductOf3Maps affine_map_xyz(affine_map_x, affine_map_y,
                                               affine_map_z);

  double xi = 0.5 * (xA + xB);
  double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);
  double eta = 0.5 * (yA + yB);
  double y = yb * (eta - yA) / (yB - yA) + ya * (yB - eta) / (yB - yA);
  double zeta = 0.5 * (zA + zB);
  double z = zb * (eta - zA) / (zB - zA) + za * (zB - eta) / (zB - zA);

  Point<3, Frame::Logical> point_A{{xA, yA, zA}};
  Point<3, Frame::Logical> point_B{{xB, yB, zB}};
  Point<3, Frame::Logical> point_xi{{xi, eta, zeta}};
  Point<3, Frame::Grid> point_a{{xa, ya, za}};
  Point<3, Frame::Grid> point_b{{xb, yb, zb}};
  Point<3, Frame::Grid> point_x{{x, y, z}};

  CHECK(affine_map_xyz(point_A) == point_a);
  CHECK(affine_map_xyz(point_B) == point_b);

  CHECK(affine_map_xyz.inverse(point_a) == point_A);
  CHECK(affine_map_xyz.inverse(point_b) == point_B);

  double invJ00 = (xB - xA) / (xb - xa);
  double invJ11 = (yB - yA) / (yb - ya);
  double invJ22 = (zB - zA) / (zb - za);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 0, 0) == invJ00);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 0, 0) == invJ00);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 0, 0) == invJ00);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 0, 1) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 0, 1) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 0, 1) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 0, 2) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 0, 2) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 0, 2) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 1, 0) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 1, 0) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 1, 0) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 1, 1) == invJ11);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 1, 1) == invJ11);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 1, 1) == invJ11);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 1, 2) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 1, 2) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 1, 2) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 2, 0) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 2, 0) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 2, 0) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 2, 1) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 2, 1) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 2, 1) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 2, 2) == invJ22);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 2, 2) == invJ22);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 2, 2) == invJ22);
}

TEST_CASE("Unit.Domain.EmbeddingMaps.Affine", "[Domain][Unit]") {
  double xA = -1.0;
  double xB = 1.0;
  double xa = -2.0;
  double xb = 2.0;

  EmbeddingMaps::AffineMap affine_map(xA, xB, xa, xb);

  double xi = 0.5 * (xA + xB);
  double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);

  Point<1, Frame::Logical> point_A(xA);
  Point<1, Frame::Logical> point_B(xB);
  Point<1, Frame::Grid> point_a(xa);
  Point<1, Frame::Grid> point_b(xb);
  Point<1, Frame::Logical> point_xi(xi);
  Point<1, Frame::Grid> point_x(x);

  CHECK(affine_map(point_A) == point_a);
  CHECK(affine_map(point_B) == point_b);
  CHECK(affine_map(point_xi) == point_x);

  CHECK(affine_map.inverse(point_a) == point_A);
  CHECK(affine_map.inverse(point_b) == point_B);
  CHECK(affine_map.inverse(point_x) == point_xi);

  double inv_jacobian_00 = (xB - xA) / (xb - xa);

  CHECK(affine_map.inv_jacobian(point_A, 0, 0) == inv_jacobian_00);
  CHECK(affine_map.inv_jacobian(point_B, 0, 0) == inv_jacobian_00);
  CHECK(affine_map.inv_jacobian(point_xi, 0, 0) == inv_jacobian_00);
}


template <size_t Dim>
void test_identity() {
  EmbeddingMaps::Identity<Dim> identity_map;
  Point<Dim, Frame::Logical> xi(1.0);
  Point<Dim, Frame::Grid> x(1.0);
  CHECK(identity_map(xi) == x);
  CHECK(identity_map.inverse(x) == xi);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      CHECK(identity_map.inv_jacobian(xi, i, j) == (i == j ? 1.0 : 0.0));
      CHECK(identity_map.jacobian(xi, i, j) == (i == j ? 1.0 : 0.0));
    }
  }
}

TEST_CASE("Unit.Domain.EmbeddingMaps.Identity", "[Domain][Unit]") {
  test_identity<1>();
  test_identity<2>();
}
