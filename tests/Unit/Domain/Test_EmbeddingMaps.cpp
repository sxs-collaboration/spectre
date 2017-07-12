// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/EmbeddingMaps/AffineMap.hpp"
#include "Domain/EmbeddingMaps/Identity.hpp"
#include "Domain/EmbeddingMaps/ProductMaps.hpp"
#include "Domain/EmbeddingMaps/Rotation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

TEST_CASE("Unit.Domain.EmbeddingMaps.ProductOf2Maps", "[Domain][Unit]") {
  const double xA = -1.0;
  const double xB = 1.0;
  const double xa = -2.0;
  const double xb = 2.0;
  EmbeddingMaps::AffineMap affine_map_x(xA, xB, xa, xb);

  const double yA = -2.0;
  const double yB = 3.0;
  const double ya = 5.0;
  const double yb = -2.0;

  EmbeddingMaps::AffineMap affine_map_y(yA, yB, ya, yb);

  EmbeddingMaps::ProductOf2Maps<1, 1> affine_map_xy(affine_map_x, affine_map_y);

  const double xi = 0.5 * (xA + xB);
  const double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);
  const double eta = 0.5 * (yA + yB);
  const double y = yb * (eta - yA) / (yB - yA) + ya * (yB - eta) / (yB - yA);

  const Point<2, Frame::Logical> point_A{{{xA, yA}}};
  const Point<2, Frame::Logical> point_B{{{xB, yB}}};
  const Point<2, Frame::Logical> point_xi{{{xi, eta}}};
  const Point<2, Frame::Grid> point_a{{{xa, ya}}};
  const Point<2, Frame::Grid> point_b{{{xb, yb}}};
  const Point<2, Frame::Grid> point_x{{{x, y}}};

  CHECK(affine_map_xy(point_A) == point_a);
  CHECK(affine_map_xy(point_B) == point_b);

  CHECK(affine_map_xy.inverse(point_a) == point_A);
  CHECK(affine_map_xy.inverse(point_b) == point_B);

  const double inv_jacobian_00 = (xB - xA) / (xb - xa);
  const double inv_jacobian_11 = (yB - yA) / (yb - ya);

  CHECK(affine_map_xy.inv_jacobian(point_A, 0, 0) == inv_jacobian_00);
  CHECK(affine_map_xy.inv_jacobian(point_B, 0, 0) == inv_jacobian_00);
  CHECK(affine_map_xy.inv_jacobian(point_xi, 0, 0) == inv_jacobian_00);

  CHECK(affine_map_xy.inv_jacobian(point_A, 0, 1) == 0.0);
  CHECK(affine_map_xy.inv_jacobian(point_B, 0, 1) == 0.0);
  CHECK(affine_map_xy.inv_jacobian(point_xi, 0, 1) == 0.0);

  CHECK(affine_map_xy.inv_jacobian(point_A, 1, 0) == 0.0);
  CHECK(affine_map_xy.inv_jacobian(point_B, 1, 0) == 0.0);
  CHECK(affine_map_xy.inv_jacobian(point_xi, 1, 0) == 0.0);

  CHECK(affine_map_xy.inv_jacobian(point_A, 1, 1) == inv_jacobian_11);
  CHECK(affine_map_xy.inv_jacobian(point_B, 1, 1) == inv_jacobian_11);
  CHECK(affine_map_xy.inv_jacobian(point_xi, 1, 1) == inv_jacobian_11);

  const double jacobian_00 = (xb - xa) / (xB - xA);
  const double jacobian_11 = (yb - ya) / (yB - yA);

  CHECK(affine_map_xy.jacobian(point_A, 0, 0) == jacobian_00);
  CHECK(affine_map_xy.jacobian(point_B, 0, 0) == jacobian_00);
  CHECK(affine_map_xy.jacobian(point_xi, 0, 0) == jacobian_00);

  CHECK(affine_map_xy.jacobian(point_A, 0, 1) == 0.0);
  CHECK(affine_map_xy.jacobian(point_B, 0, 1) == 0.0);
  CHECK(affine_map_xy.jacobian(point_xi, 0, 1) == 0.0);

  CHECK(affine_map_xy.jacobian(point_A, 1, 0) == 0.0);
  CHECK(affine_map_xy.jacobian(point_B, 1, 0) == 0.0);
  CHECK(affine_map_xy.jacobian(point_xi, 1, 0) == 0.0);

  CHECK(affine_map_xy.jacobian(point_A, 1, 1) == jacobian_11);
  CHECK(affine_map_xy.jacobian(point_B, 1, 1) == jacobian_11);
  CHECK(affine_map_xy.jacobian(point_xi, 1, 1) == jacobian_11);

  auto affine_map_xy_clone = affine_map_xy.get_clone();
  CHECK(affine_map_xy_clone.get() != &affine_map_xy);
  CHECK((*affine_map_xy_clone)(point_A) == point_a);
}

TEST_CASE("Unit.Domain.EmbeddingMaps.ProductOf3Maps", "[Domain][Unit]") {
  const double xA = -1.0;
  const double xB = 1.0;
  const double xa = -2.0;
  const double xb = 2.0;
  EmbeddingMaps::AffineMap affine_map_x(xA, xB, xa, xb);

  const double yA = -2.0;
  const double yB = 3.0;
  const double ya = 5.0;
  const double yb = -2.0;

  EmbeddingMaps::AffineMap affine_map_y(yA, yB, ya, yb);

  const double zA = 4.0;
  const double zB = 8.0;
  const double za = -15.0;
  const double zb = 12.0;

  EmbeddingMaps::AffineMap affine_map_z(zA, zB, za, zb);

  EmbeddingMaps::ProductOf3Maps affine_map_xyz(affine_map_x, affine_map_y,
                                               affine_map_z);

  const double xi = 0.5 * (xA + xB);
  const double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);
  const double eta = 0.5 * (yA + yB);
  const double y = yb * (eta - yA) / (yB - yA) + ya * (yB - eta) / (yB - yA);
  const double zeta = 0.5 * (zA + zB);
  const double z = zb * (eta - zA) / (zB - zA) + za * (zB - eta) / (zB - zA);

  const Point<3, Frame::Logical> point_A{{{xA, yA, zA}}};
  const Point<3, Frame::Logical> point_B{{{xB, yB, zB}}};
  const Point<3, Frame::Logical> point_xi{{{xi, eta, zeta}}};
  const Point<3, Frame::Grid> point_a{{{xa, ya, za}}};
  const Point<3, Frame::Grid> point_b{{{xb, yb, zb}}};
  const Point<3, Frame::Grid> point_x{{{x, y, z}}};

  CHECK(affine_map_xyz(point_A) == point_a);
  CHECK(affine_map_xyz(point_B) == point_b);

  CHECK(affine_map_xyz.inverse(point_a) == point_A);
  CHECK(affine_map_xyz.inverse(point_b) == point_B);

  const double inv_jacobian_00 = (xB - xA) / (xb - xa);
  const double inv_jacobian_11 = (yB - yA) / (yb - ya);
  const double inv_jacobian_22 = (zB - zA) / (zb - za);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 0, 0) == inv_jacobian_00);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 0, 0) == inv_jacobian_00);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 0, 0) == inv_jacobian_00);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 0, 1) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 0, 1) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 0, 1) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 0, 2) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 0, 2) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 0, 2) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 1, 0) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 1, 0) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 1, 0) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 1, 1) == inv_jacobian_11);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 1, 1) == inv_jacobian_11);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 1, 1) == inv_jacobian_11);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 1, 2) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 1, 2) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 1, 2) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 2, 0) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 2, 0) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 2, 0) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 2, 1) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 2, 1) == 0.0);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 2, 1) == 0.0);

  CHECK(affine_map_xyz.inv_jacobian(point_A, 2, 2) == inv_jacobian_22);
  CHECK(affine_map_xyz.inv_jacobian(point_B, 2, 2) == inv_jacobian_22);
  CHECK(affine_map_xyz.inv_jacobian(point_xi, 2, 2) == inv_jacobian_22);

  const double jacobian_00 = (xb - xa) / (xB - xA);
  const double jacobian_11 = (yb - ya) / (yB - yA);
  const double jacobian_22 = (zb - za) / (zB - zA);

  CHECK(affine_map_xyz.jacobian(point_A, 0, 0) == jacobian_00);
  CHECK(affine_map_xyz.jacobian(point_B, 0, 0) == jacobian_00);
  CHECK(affine_map_xyz.jacobian(point_xi, 0, 0) == jacobian_00);

  CHECK(affine_map_xyz.jacobian(point_A, 0, 1) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_B, 0, 1) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_xi, 0, 1) == 0.0);

  CHECK(affine_map_xyz.jacobian(point_A, 0, 2) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_B, 0, 2) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_xi, 0, 2) == 0.0);

  CHECK(affine_map_xyz.jacobian(point_A, 1, 0) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_B, 1, 0) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_xi, 1, 0) == 0.0);

  CHECK(affine_map_xyz.jacobian(point_A, 1, 1) == jacobian_11);
  CHECK(affine_map_xyz.jacobian(point_B, 1, 1) == jacobian_11);
  CHECK(affine_map_xyz.jacobian(point_xi, 1, 1) == jacobian_11);

  CHECK(affine_map_xyz.jacobian(point_A, 1, 2) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_B, 1, 2) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_xi, 1, 2) == 0.0);

  CHECK(affine_map_xyz.jacobian(point_A, 2, 0) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_B, 2, 0) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_xi, 2, 0) == 0.0);

  CHECK(affine_map_xyz.jacobian(point_A, 2, 1) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_B, 2, 1) == 0.0);
  CHECK(affine_map_xyz.jacobian(point_xi, 2, 1) == 0.0);

  CHECK(affine_map_xyz.jacobian(point_A, 2, 2) == jacobian_22);
  CHECK(affine_map_xyz.jacobian(point_B, 2, 2) == jacobian_22);
  CHECK(affine_map_xyz.jacobian(point_xi, 2, 2) == jacobian_22);

  auto affine_map_xyz_clone = affine_map_xyz.get_clone();
  CHECK(affine_map_xyz_clone.get() != &affine_map_xyz);
  CHECK((*affine_map_xyz_clone)(point_A) == point_a);
}

TEST_CASE("Unit.Domain.EmbeddingMaps.Rotation<2>", "[Domain][Unit]") {
  EmbeddingMaps::Rotation<2> half_pi_rotation_map(M_PI_2);

  const Point<2, Frame::Logical> xi0{{{0.0, 0.0}}};
  const Point<2, Frame::Grid> x0{{{0.0, 0.0}}};

  CHECK(half_pi_rotation_map(xi0)[0] == Approx(x0[0]));
  CHECK(half_pi_rotation_map(xi0)[1] == Approx(x0[1]));

  CHECK(half_pi_rotation_map.inverse(x0)[0] == Approx(xi0[0]));
  CHECK(half_pi_rotation_map.inverse(x0)[1] == Approx(xi0[1]));

  const Point<2, Frame::Logical> xi1{{{1.0, 0.0}}};
  const Point<2, Frame::Grid> x1{{{0.0, 1.0}}};

  CHECK(half_pi_rotation_map(xi1)[0] == Approx(x1[0]));
  CHECK(half_pi_rotation_map(xi1)[1] == Approx(x1[1]));

  CHECK(half_pi_rotation_map.inverse(x1)[0] == Approx(xi1[0]));
  CHECK(half_pi_rotation_map.inverse(x1)[1] == Approx(xi1[1]));

  const Point<2, Frame::Logical> xi2{{{0.0, 1.0}}};
  const Point<2, Frame::Grid> x2{{{-1.0, 0.0}}};

  CHECK(half_pi_rotation_map(xi2)[0] == Approx(x2[0]));
  CHECK(half_pi_rotation_map(xi2)[1] == Approx(x2[1]));

  CHECK(half_pi_rotation_map.inverse(x2)[0] == Approx(xi2[0]));
  CHECK(half_pi_rotation_map.inverse(x2)[1] == Approx(xi2[1]));

  CHECK(half_pi_rotation_map.inv_jacobian(xi2, 0, 0) == Approx(0.0));
  CHECK(half_pi_rotation_map.inv_jacobian(xi2, 0, 1) == Approx(1.0));
  CHECK(half_pi_rotation_map.inv_jacobian(xi2, 1, 0) == Approx(-1.0));
  CHECK(half_pi_rotation_map.inv_jacobian(xi2, 1, 1) == Approx(0.0));

  CHECK(half_pi_rotation_map.jacobian(xi2, 0, 0) == Approx(0.0));
  CHECK(half_pi_rotation_map.jacobian(xi2, 0, 1) == Approx(-1.0));
  CHECK(half_pi_rotation_map.jacobian(xi2, 1, 0) == Approx(1.0));
  CHECK(half_pi_rotation_map.jacobian(xi2, 1, 1) == Approx(0.0));

  auto half_pi_rotation_map_clone = half_pi_rotation_map.get_clone();
  CHECK(half_pi_rotation_map_clone.get() != &half_pi_rotation_map);
  for (size_t i = 0; i < 2; ++i) {
    CHECK((*half_pi_rotation_map_clone)(xi1)[i] == Approx(x1[i]));
  }
}

void test_rotation_3(const EmbeddingMaps::Rotation<3>& three_dim_rotation_map,
                     const Point<3, Frame::Grid>& xi_hat,
                     const Point<3, Frame::Grid>& eta_hat,
                     const Point<3, Frame::Grid>& zeta_hat) {
  const Point<3, Frame::Logical> zero_logical{{{0.0, 0.0, 0.0}}};
  const Point<3, Frame::Grid> zero_grid{{{0.0, 0.0, 0.0}}};
  const Point<3, Frame::Logical> xi{{{1.0, 0.0, 0.0}}};
  const Point<3, Frame::Logical> eta{{{0.0, 1.0, 0.0}}};
  const Point<3, Frame::Logical> zeta{{{0.0, 0.0, 1.0}}};
  for (size_t i = 0; i < 3; ++i) {
    CHECK(three_dim_rotation_map(zero_logical)[i] == Approx(zero_grid[i]));
    CHECK(three_dim_rotation_map(xi)[i] == Approx(xi_hat[i]));
    CHECK(three_dim_rotation_map(eta)[i] == Approx(eta_hat[i]));
    CHECK(three_dim_rotation_map(zeta)[i] == Approx(zeta_hat[i]));
    CHECK(three_dim_rotation_map.inverse(zero_grid)[i] ==
          Approx(zero_logical[i]));
    CHECK(three_dim_rotation_map.inverse(xi_hat)[i] == Approx(xi[i]));
    CHECK(three_dim_rotation_map.inverse(eta_hat)[i] == Approx(eta[i]));
    CHECK(three_dim_rotation_map.inverse(zeta_hat)[i] == Approx(zeta[i]));
    CHECK(three_dim_rotation_map.inv_jacobian(zero_logical, 0, i) ==
          Approx(xi_hat[i]));
    CHECK(three_dim_rotation_map.inv_jacobian(zero_logical, 1, i) ==
          Approx(eta_hat[i]));
    CHECK(three_dim_rotation_map.inv_jacobian(zero_logical, 2, i) ==
          Approx(zeta_hat[i]));
    CHECK(three_dim_rotation_map.jacobian(zero_logical, i, 0) ==
          Approx(xi_hat[i]));
    CHECK(three_dim_rotation_map.jacobian(zero_logical, i, 1) ==
          Approx(eta_hat[i]));
    CHECK(three_dim_rotation_map.jacobian(zero_logical, i, 2) ==
          Approx(zeta_hat[i]));
  }
  auto rotated_3_clone = three_dim_rotation_map.get_clone();
  CHECK(rotated_3_clone.get() != &three_dim_rotation_map);
  for (size_t i = 0; i < 3; ++i) {
    CHECK((*rotated_3_clone)(xi)[i] == Approx(xi_hat[i]));
  }
}

TEST_CASE("Unit.Domain.EmbeddingMaps.Rotation<3>", "[Domain][Unit]") {
  test_rotation_3(EmbeddingMaps::Rotation<3>(0.0, 0.0, 0.0),
                  Point<3, Frame::Grid>{{{1.0, 0.0, 0.0}}},
                  Point<3, Frame::Grid>{{{0.0, 1.0, 0.0}}},
                  Point<3, Frame::Grid>{{{0.0, 0.0, 1.0}}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(M_PI_2, 0.0, 0.0),
                  Point<3, Frame::Grid>{{{0.0, 1.0, 0.0}}},
                  Point<3, Frame::Grid>{{{-1.0, 0.0, 0.0}}},
                  Point<3, Frame::Grid>{{{0.0, 0.0, 1.0}}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(M_PI, 0.0, 0.0),
                  Point<3, Frame::Grid>{{{-1.0, 0.0, 0.0}}},
                  Point<3, Frame::Grid>{{{0.0, -1.0, 0.0}}},
                  Point<3, Frame::Grid>{{{0.0, 0.0, 1.0}}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(-M_PI_2, 0.0, 0.0),
                  Point<3, Frame::Grid>{{{0.0, -1.0, 0.0}}},
                  Point<3, Frame::Grid>{{{1.0, 0.0, 0.0}}},
                  Point<3, Frame::Grid>{{{0.0, 0.0, 1.0}}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(0.0, -M_PI_2, 0.0),
                  Point<3, Frame::Grid>{{{0.0, 0.0, 1.0}}},
                  Point<3, Frame::Grid>{{{0.0, 1.0, 0.0}}},
                  Point<3, Frame::Grid>{{{-1.0, 0.0, 0.0}}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(0.0, M_PI_2, 0.0),
                  Point<3, Frame::Grid>{{{0.0, 0.0, -1.0}}},
                  Point<3, Frame::Grid>{{{0.0, 1.0, 0.0}}},
                  Point<3, Frame::Grid>{{{1.0, 0.0, 0.0}}});
}

TEST_CASE("Unit.Domain.EmbeddingMaps.Affine", "[Domain][Unit]") {
  const double xA = -1.0;
  const double xB = 1.0;
  const double xa = -2.0;
  const double xb = 2.0;

  EmbeddingMaps::AffineMap affine_map(xA, xB, xa, xb);

  const double xi = 0.5 * (xA + xB);
  const double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);

  const Point<1, Frame::Logical> point_A(xA);
  const Point<1, Frame::Logical> point_B(xB);
  const Point<1, Frame::Grid> point_a(xa);
  const Point<1, Frame::Grid> point_b(xb);
  const Point<1, Frame::Logical> point_xi(xi);
  const Point<1, Frame::Grid> point_x(x);

  CHECK(affine_map(point_A) == point_a);
  CHECK(affine_map(point_B) == point_b);
  CHECK(affine_map(point_xi) == point_x);

  CHECK(affine_map.inverse(point_a) == point_A);
  CHECK(affine_map.inverse(point_b) == point_B);
  CHECK(affine_map.inverse(point_x) == point_xi);

  const double inv_jacobian_00 = (xB - xA) / (xb - xa);

  CHECK(affine_map.inv_jacobian(point_A, 0, 0) == inv_jacobian_00);
  CHECK(affine_map.inv_jacobian(point_B, 0, 0) == inv_jacobian_00);
  CHECK(affine_map.inv_jacobian(point_xi, 0, 0) == inv_jacobian_00);

  const double jacobian_00 = (xb - xa) / (xB - xA);
  CHECK(affine_map.jacobian(point_A, 0, 0) == jacobian_00);
  CHECK(affine_map.jacobian(point_B, 0, 0) == jacobian_00);
  CHECK(affine_map.jacobian(point_xi, 0, 0) == jacobian_00);

  auto affine_map_clone = affine_map.get_clone();
  CHECK(affine_map_clone.get() != &affine_map);
  CHECK((*affine_map_clone)(point_xi) == point_x);
}

template <size_t Dim>
void test_identity() {
  EmbeddingMaps::Identity<Dim> identity_map;
  const Point<Dim, Frame::Logical> xi(1.0);
  const Point<Dim, Frame::Grid> x(1.0);
  CHECK(identity_map(xi) == x);
  CHECK(identity_map.inverse(x) == xi);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      CHECK(identity_map.inv_jacobian(xi, i, j) == (i == j ? 1.0 : 0.0));
      CHECK(identity_map.jacobian(xi, i, j) == (i == j ? 1.0 : 0.0));
    }
  }
  auto clone_identity = identity_map.get_clone();
  CHECK(clone_identity.get() != &identity_map);
  CHECK((*clone_identity)(xi) == x);
}

TEST_CASE("Unit.Domain.EmbeddingMaps.Identity", "[Domain][Unit]") {
  test_identity<1>();
  test_identity<2>();
}
