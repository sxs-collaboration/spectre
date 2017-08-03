// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/EmbeddingMaps/AffineMap.hpp"
#include "Domain/EmbeddingMaps/CoordinateMap.hpp"
#include "Domain/EmbeddingMaps/Identity.hpp"
#include "Domain/EmbeddingMaps/ProductMaps.hpp"
#include "Domain/EmbeddingMaps/Rotation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.EmbeddingMaps.ProductOf2Maps",
                  "[Domain][Unit]") {
  using affine_map = EmbeddingMaps::AffineMap;
  using affine_map_2d = EmbeddingMaps::ProductOf2Maps<affine_map, affine_map>;

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
}

SPECTRE_TEST_CASE("Unit.Domain.EmbeddingMaps.ProductOf3Maps",
                  "[Domain][Unit]") {
  using affine_map = EmbeddingMaps::AffineMap;
  using affine_map_3d =
      EmbeddingMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;

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
}

SPECTRE_TEST_CASE("Unit.Domain.EmbeddingMaps.Rotation<2>", "[Domain][Unit]") {
  EmbeddingMaps::Rotation<2> half_pi_rotation_map(M_PI_2);
  Approx approx = Approx::custom().epsilon(1e-15);

  const auto xi0 = make_array<2>(0.0);
  const auto x0 = make_array<2>(0.0);

  CHECK(half_pi_rotation_map(xi0)[0] == approx(x0[0]));
  CHECK(half_pi_rotation_map(xi0)[1] == approx(x0[1]));

  CHECK(half_pi_rotation_map.inverse(x0)[0] == approx(xi0[0]));
  CHECK(half_pi_rotation_map.inverse(x0)[1] == approx(xi0[1]));

  const std::array<double, 2> xi1{{1.0, 0.0}};
  const std::array<double, 2> x1{{0.0, 1.0}};

  CHECK(half_pi_rotation_map(xi1)[0] == approx(x1[0]));
  CHECK(half_pi_rotation_map(xi1)[1] == approx(x1[1]));

  CHECK(half_pi_rotation_map.inverse(x1)[0] == approx(xi1[0]));
  CHECK(half_pi_rotation_map.inverse(x1)[1] == approx(xi1[1]));

  const std::array<double, 2> xi2{{0.0, 1.0}};
  const std::array<double, 2> x2{{-1.0, 0.0}};

  CHECK(half_pi_rotation_map(xi2)[0] == approx(x2[0]));
  CHECK(half_pi_rotation_map(xi2)[1] == approx(x2[1]));

  CHECK(half_pi_rotation_map.inverse(x2)[0] == approx(xi2[0]));
  CHECK(half_pi_rotation_map.inverse(x2)[1] == approx(xi2[1]));

  const auto inv_jac = half_pi_rotation_map.inv_jacobian(xi2);
  CHECK((inv_jac.template get<0, 0>()) == approx(0.0));
  CHECK((inv_jac.template get<0, 1>()) == approx(1.0));
  CHECK((inv_jac.template get<1, 0>()) == approx(-1.0));
  CHECK((inv_jac.template get<1, 1>()) == approx(0.0));

  const auto jac = half_pi_rotation_map.jacobian(xi2);
  CHECK((jac.template get<0, 0>()) == approx(0.0));
  CHECK((jac.template get<0, 1>()) == approx(-1.0));
  CHECK((jac.template get<1, 0>()) == approx(1.0));
  CHECK((jac.template get<1, 1>()) == approx(0.0));

  // Check inequivalence operator
  CHECK_FALSE(half_pi_rotation_map != half_pi_rotation_map);
  CHECK(half_pi_rotation_map ==
        serialize_and_deserialize(half_pi_rotation_map));
}

template <typename T>
void test_rotation_3(const EmbeddingMaps::Rotation<3>& three_dim_rotation_map,
                     const std::array<T, 3>& xi_hat,
                     const std::array<T, 3>& eta_hat,
                     const std::array<T, 3>& zeta_hat) {
  Approx approx = Approx::custom().epsilon(1e-15);

  const std::array<T, 3> zero_logical{{0.0, 0.0, 0.0}};
  const std::array<T, 3> zero_grid{{0.0, 0.0, 0.0}};
  const std::array<T, 3> xi{{1.0, 0.0, 0.0}};
  const std::array<T, 3> eta{{0.0, 1.0, 0.0}};
  const std::array<T, 3> zeta{{0.0, 0.0, 1.0}};

  const auto inv_jac = three_dim_rotation_map.inv_jacobian(zero_logical);
  const auto jac = three_dim_rotation_map.jacobian(zero_logical);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(gsl::at(three_dim_rotation_map(zero_logical), i) ==
          approx(gsl::at(zero_grid, i)));
    CHECK(gsl::at(three_dim_rotation_map(xi), i) == approx(gsl::at(xi_hat, i)));
    CHECK(gsl::at(three_dim_rotation_map(eta), i) ==
          approx(gsl::at(eta_hat, i)));
    CHECK(gsl::at(three_dim_rotation_map(zeta), i) ==
          approx(gsl::at(zeta_hat, i)));
    CHECK(gsl::at(three_dim_rotation_map.inverse(zero_grid), i) ==
          approx(gsl::at(zero_logical, i)));
    CHECK(gsl::at(three_dim_rotation_map.inverse(xi_hat), i) ==
          approx(gsl::at(xi, i)));
    CHECK(gsl::at(three_dim_rotation_map.inverse(eta_hat), i) ==
          approx(gsl::at(eta, i)));
    CHECK(gsl::at(three_dim_rotation_map.inverse(zeta_hat), i) ==
          approx(gsl::at(zeta, i)));
    CHECK(inv_jac.get(0, i) == approx(gsl::at(xi_hat, i)));
    CHECK(inv_jac.get(1, i) == approx(gsl::at(eta_hat, i)));
    CHECK(inv_jac.get(2, i) == approx(gsl::at(zeta_hat, i)));
    CHECK(jac.get(i, 0) == approx(gsl::at(xi_hat, i)));
    CHECK(jac.get(i, 1) == approx(gsl::at(eta_hat, i)));
    CHECK(jac.get(i, 2) == approx(gsl::at(zeta_hat, i)));
  }
  // Check inequivalence operator
  CHECK_FALSE(three_dim_rotation_map != three_dim_rotation_map);
  CHECK(three_dim_rotation_map ==
        serialize_and_deserialize(three_dim_rotation_map));
}

SPECTRE_TEST_CASE("Unit.Domain.EmbeddingMaps.Rotation<3>", "[Domain][Unit]") {
  test_rotation_3(EmbeddingMaps::Rotation<3>(0.0, 0.0, 0.0),
                  std::array<double, 3>{{1.0, 0.0, 0.0}},
                  std::array<double, 3>{{0.0, 1.0, 0.0}},
                  std::array<double, 3>{{0.0, 0.0, 1.0}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(M_PI_2, 0.0, 0.0),
                  std::array<double, 3>{{0.0, 1.0, 0.0}},
                  std::array<double, 3>{{-1.0, 0.0, 0.0}},
                  std::array<double, 3>{{0.0, 0.0, 1.0}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(M_PI, 0.0, 0.0),
                  std::array<double, 3>{{-1.0, 0.0, 0.0}},
                  std::array<double, 3>{{0.0, -1.0, 0.0}},
                  std::array<double, 3>{{0.0, 0.0, 1.0}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(-M_PI_2, 0.0, 0.0),
                  std::array<double, 3>{{0.0, -1.0, 0.0}},
                  std::array<double, 3>{{1.0, 0.0, 0.0}},
                  std::array<double, 3>{{0.0, 0.0, 1.0}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(0.0, -M_PI_2, 0.0),
                  std::array<double, 3>{{0.0, 0.0, 1.0}},
                  std::array<double, 3>{{0.0, 1.0, 0.0}},
                  std::array<double, 3>{{-1.0, 0.0, 0.0}});

  test_rotation_3(EmbeddingMaps::Rotation<3>(0.0, M_PI_2, 0.0),
                  std::array<double, 3>{{0.0, 0.0, -1.0}},
                  std::array<double, 3>{{0.0, 1.0, 0.0}},
                  std::array<double, 3>{{1.0, 0.0, 0.0}});
}

SPECTRE_TEST_CASE("Unit.Domain.EmbeddingMaps.Affine", "[Domain][Unit]") {
  const double xA = -1.0;
  const double xB = 1.0;
  const double xa = -2.0;
  const double xb = 2.0;

  EmbeddingMaps::AffineMap affine_map(xA, xB, xa, xb);

  const double xi = 0.5 * (xA + xB);
  const double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);

  const std::array<double, 1> point_A{{xA}};
  const std::array<double, 1> point_B{{xB}};
  const std::array<double, 1> point_a{{xa}};
  const std::array<double, 1> point_b{{xb}};
  const std::array<double, 1> point_xi{{xi}};
  const std::array<double, 1> point_x{{x}};

  CHECK(affine_map(point_A) == point_a);
  CHECK(affine_map(point_B) == point_b);
  CHECK(affine_map(point_xi) == point_x);

  CHECK(affine_map.inverse(point_a) == point_A);
  CHECK(affine_map.inverse(point_b) == point_B);
  CHECK(affine_map.inverse(point_x) == point_xi);

  const double inv_jacobian_00 = (xB - xA) / (xb - xa);

  CHECK((affine_map.inv_jacobian(point_A).template get<0, 0>()) ==
        inv_jacobian_00);
  CHECK((affine_map.inv_jacobian(point_B).template get<0, 0>()) ==
        inv_jacobian_00);
  CHECK((affine_map.inv_jacobian(point_xi).template get<0, 0>()) ==
        inv_jacobian_00);

  const double jacobian_00 = (xb - xa) / (xB - xA);
  CHECK((affine_map.jacobian(point_A).template get<0, 0>()) == jacobian_00);
  CHECK((affine_map.jacobian(point_B).template get<0, 0>()) == jacobian_00);
  CHECK((affine_map.jacobian(point_xi).template get<0, 0>()) == jacobian_00);

  // Check inequivalence operator
  CHECK_FALSE(affine_map != affine_map);
  CHECK(affine_map == serialize_and_deserialize(affine_map));
}

template <size_t Dim>
void test_identity() {
  EmbeddingMaps::Identity<Dim> identity_map;
  const auto xi = make_array<Dim>(1.0);
  const auto x = make_array<Dim>(1.0);
  CHECK(identity_map(xi) == x);
  CHECK(identity_map.inverse(x) == xi);
  const auto inv_jac = identity_map.inv_jacobian(xi);
  const auto jac = identity_map.jacobian(xi);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      CHECK(inv_jac.get(i, j) == (i == j ? 1.0 : 0.0));
      CHECK(jac.get(i, j) == (i == j ? 1.0 : 0.0));
    }
  }
  // Checks the inequivalence operator
  CHECK_FALSE(identity_map != identity_map);
  CHECK(identity_map == serialize_and_deserialize(identity_map));
}

SPECTRE_TEST_CASE("Unit.Domain.EmbeddingMaps.Identity", "[Domain][Unit]") {
  test_identity<1>();
  test_identity<2>();
}

namespace {
void test_single_coordinate_map() {
  using affine_map1d = EmbeddingMaps::AffineMap;

  const auto affine1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map1d{-1.0, 1.0, 2.0, 8.0});
  const auto first_affine1d = affine_map1d{-1.0, 1.0, 2.0, 8.0};

  std::array<std::array<double, 1>, 4> coords1d{
      {{{0.1}}, {{-8.2}}, {{5.7}}, {{2.9}}}};

  for (const auto& coord : coords1d) {
    CHECK((make_array<double, 1>(affine1d(Point<1, Frame::Logical>{
              {{coord[0]}}}))) == first_affine1d(coord));
    CHECK((make_array<double, 1>(affine1d.inverse(Point<1, Frame::Grid>{
              {{coord[0]}}}))) == first_affine1d.inverse(coord));

    const auto jac = affine1d.jacobian(Point<1, Frame::Logical>{{{coord[0]}}});
    const auto expected_jac = first_affine1d.jacobian(coord);
    CHECK(jac.get(0, 0) == expected_jac.get(0, 0));

    const auto inv_jac =
        affine1d.inv_jacobian(Point<1, Frame::Logical>{{{coord[0]}}});
    const auto expected_inv_jac = first_affine1d.inv_jacobian(coord);
    CHECK(inv_jac.get(0, 0) == expected_inv_jac.get(0, 0));
  }

  using rotate2d = EmbeddingMaps::Rotation<2>;

  const auto rotated2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotate2d{M_PI_4});
  const auto first_rotated2d = rotate2d{M_PI_4};

  std::array<std::array<double, 2>, 4> coords2d{
      {{{0.1, 2.8}}, {{-8.2, 2.8}}, {{5.7, -4.9}}, {{2.9, 3.4}}}};

  for (const auto& coord : coords2d) {
    CHECK((make_array<double, 2>(rotated2d(Point<2, Frame::Logical>{
              {{coord[0], coord[1]}}}))) == first_rotated2d(coord));
    CHECK((make_array<double, 2>(rotated2d.inverse(Point<2, Frame::Grid>{
              {{coord[0], coord[1]}}}))) == first_rotated2d.inverse(coord));

    const auto jac =
        rotated2d.jacobian(Point<2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto expected_jac = first_rotated2d.jacobian(coord);
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = rotated2d.inv_jacobian(
        Point<2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto expected_inv_jac = first_rotated2d.inv_jacobian(coord);
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }

  using rotate3d = EmbeddingMaps::Rotation<3>;

  const auto rotated3d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      rotate3d{M_PI_4, M_PI_4, M_PI_2});
  const auto first_rotated3d = rotate3d{M_PI_4, M_PI_4, M_PI_2};

  std::array<std::array<double, 3>, 4> coords3d{{{{0.1, 2.8, 9.3}},
                                                 {{-8.2, 2.8, -9.7}},
                                                 {{5.7, -4.9, 8.1}},
                                                 {{2.9, 3.4, -7.8}}}};

  for (const auto& coord : coords3d) {
    CHECK((make_array<double, 3>(rotated3d(Point<3, Frame::Logical>{
              {{coord[0], coord[1], coord[2]}}}))) == first_rotated3d(coord));
    CHECK((make_array<double, 3>(rotated3d.inverse(
              Point<3, Frame::Grid>{{{coord[0], coord[1], coord[2]}}}))) ==
          first_rotated3d.inverse(coord));

    const auto jac = rotated3d.jacobian(
        Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto expected_jac = first_rotated3d.jacobian(coord);
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = rotated3d.inv_jacobian(
        Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto expected_inv_jac = first_rotated3d.inv_jacobian(coord);
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }
}

void test_coordinate_map_with_affine_map() {
  using affine_map = EmbeddingMaps::AffineMap;
  using affine_map_2d = EmbeddingMaps::ProductOf2Maps<affine_map, affine_map>;
  using affine_map_3d =
      EmbeddingMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;

  constexpr size_t number_of_points_checked = 10;
  Approx approx = Approx::custom().epsilon(1.0e-14);

  // Test 1D
  const auto map = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map{-1.0, 1.0, 0.0, 2.3}, affine_map{0.0, 2.3, -0.5, 0.5});
  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    CHECK((Point<1, Frame::Grid>(1.0 / i + -0.5))[0] ==
          approx(map(Point<1, Frame::Logical>{2.0 / i + -1.0})[0]));
    CHECK((Point<1, Frame::Logical>(2.0 / i + -1.0))[0] ==
          approx(map.inverse(Point<1, Frame::Grid>{1.0 / i + -0.5})[0]));

    CHECK(approx(map.inv_jacobian(Point<1, Frame::Logical>{2.0 / i + -1.0})
                     .get(0, 0)) == 2.0);
    CHECK(approx(map.jacobian(Point<1, Frame::Logical>{2.0 / i + -1.0})
                     .get(0, 0)) == 0.5);
  }

  // Test 2D
  const auto prod_map2d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map_2d{affine_map{-1.0, 1.0, 0.0, 2.0},
                    affine_map{0.0, 2.0, -0.5, 0.5}},
      affine_map_2d{affine_map{0.0, 2.0, 2.0, 6.0},
                    affine_map{-0.5, 0.5, 0.0, 8.0}});
  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    const auto mapped_point =
        prod_map2d(Point<2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    const auto expected_mapped_point =
        Point<2, Frame::Grid>{{{4.0 / i + 2.0, 8.0 / i + 0.0}}};
    CHECK(expected_mapped_point.template get<0>() ==
          approx(mapped_point.template get<0>()));
    CHECK(expected_mapped_point.template get<1>() ==
          approx(mapped_point.template get<1>()));

    const auto inv_mapped_point = prod_map2d.inverse(
        Point<2, Frame::Grid>{{{4.0 / i + 2.0, 8.0 / i + 0.0}}});
    const auto expected_inv_mapped_point =
        Point<2, Frame::Grid>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}};
    CHECK(expected_inv_mapped_point.template get<0>() ==
          approx(inv_mapped_point.template get<0>()));
    CHECK(expected_inv_mapped_point.template get<1>() ==
          approx(inv_mapped_point.template get<1>()));

    const auto inv_jac = prod_map2d.inv_jacobian(
        Point<2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    CHECK(0.5 == approx(inv_jac.template get<0, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<1, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<0, 1>()));
    CHECK(0.25 == approx(inv_jac.template get<1, 1>()));

    const auto jac = prod_map2d.jacobian(
        Point<2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    CHECK(2.0 == approx(jac.template get<0, 0>()));
    CHECK(0.0 == approx(jac.template get<1, 0>()));
    CHECK(0.0 == approx(jac.template get<0, 1>()));
    CHECK(4.0 == approx(jac.template get<1, 1>()));
  }

  // Test 3D
  const auto prod_map3d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map_3d{affine_map{-1.0, 1.0, 0.0, 2.0},
                    affine_map{0.0, 2.0, -0.5, 0.5},
                    affine_map{5.0, 7.0, -7.0, 7.0}},
      affine_map_3d{affine_map{0.0, 2.0, 2.0, 6.0},
                    affine_map{-0.5, 0.5, 0.0, 8.0},
                    affine_map{-7.0, 7.0, 3.0, 23.0}});

  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    const auto mapped_point = prod_map3d(Point<3, Frame::Logical>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    const auto expected_mapped_point =
        Point<3, Frame::Grid>{{{4.0 / i + 2.0, 8.0 / i + 0.0, 3.0 + 20.0 / i}}};
    CHECK(expected_mapped_point.template get<0>() ==
          approx(mapped_point.template get<0>()));
    CHECK(expected_mapped_point.template get<1>() ==
          approx(mapped_point.template get<1>()));
    CHECK(expected_mapped_point.template get<2>() ==
          approx(mapped_point.template get<2>()));

    const auto inv_mapped_point = prod_map3d.inverse(Point<3, Frame::Grid>{
        {{4.0 / i + 2.0, 8.0 / i + 0.0, 3.0 + 20.0 / i}}});
    const auto expected_inv_mapped_point =
        Point<3, Frame::Grid>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}};
    CHECK(expected_inv_mapped_point.template get<0>() ==
          approx(inv_mapped_point.template get<0>()));
    CHECK(expected_inv_mapped_point.template get<1>() ==
          approx(inv_mapped_point.template get<1>()));
    CHECK(expected_inv_mapped_point.template get<2>() ==
          approx(inv_mapped_point.template get<2>()));

    const auto inv_jac = prod_map3d.inv_jacobian(Point<3, Frame::Logical>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    CHECK(0.5 == approx(inv_jac.template get<0, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<1, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<0, 1>()));
    CHECK(0.25 == approx(inv_jac.template get<1, 1>()));
    CHECK(0.0 == approx(inv_jac.template get<0, 2>()));
    CHECK(0.0 == approx(inv_jac.template get<1, 2>()));
    CHECK(0.0 == approx(inv_jac.template get<2, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<2, 1>()));
    CHECK(0.1 == approx(inv_jac.template get<2, 2>()));

    const auto jac = prod_map3d.jacobian(Point<3, Frame::Logical>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    CHECK(2.0 == approx(jac.template get<0, 0>()));
    CHECK(0.0 == approx(jac.template get<1, 0>()));
    CHECK(0.0 == approx(jac.template get<0, 1>()));
    CHECK(4.0 == approx(jac.template get<1, 1>()));
    CHECK(0.0 == approx(jac.template get<0, 2>()));
    CHECK(0.0 == approx(jac.template get<1, 2>()));
    CHECK(0.0 == approx(jac.template get<2, 0>()));
    CHECK(0.0 == approx(jac.template get<2, 1>()));
    CHECK(10.0 == approx(jac.template get<2, 2>()));
  }
}

void test_coordinate_map_with_rotation_map() {
  using rotate2d = EmbeddingMaps::Rotation<2>;
  using rotate3d = EmbeddingMaps::Rotation<3>;

  // No 1D test because it would just the be affine map test

  // Test 2D
  const auto double_rotated2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotate2d{M_PI_4},
                                                       rotate2d{M_PI_2});
  const auto first_rotated2d = rotate2d{M_PI_4};
  const auto second_rotated2d = rotate2d{M_PI_2};

  std::array<std::array<double, 2>, 4> coords2d{
      {{{0.1, 2.8}}, {{-8.2, 2.8}}, {{5.7, -4.9}}, {{2.9, 3.4}}}};

  for (size_t i = 0; i < coords2d.size(); ++i) {
    const auto coord = gsl::at(coords2d, i);
    CHECK((make_array<double, 2>(double_rotated2d(
              Point<2, Frame::Logical>{{{coord[0], coord[1]}}}))) ==
          second_rotated2d(first_rotated2d(coord)));
    CHECK((make_array<double, 2>(double_rotated2d.inverse(
              Point<2, Frame::Grid>{{{coord[0], coord[1]}}}))) ==
          first_rotated2d.inverse(second_rotated2d.inverse(coord)));

    const auto jac = double_rotated2d.jacobian(
        Point<2, Frame::Logical>{{{coord[0], coord[1]}}});

    const auto expected_jac = [&first_rotated2d, &second_rotated2d,
                               &coords2d](const size_t ii) {

      const auto first_jac = first_rotated2d.jacobian(gsl::at(coords2d, ii));
      auto second_jac =
          second_rotated2d.jacobian(first_rotated2d(gsl::at(coords2d, ii)));

      std::array<double, 2> temp{};
      for (size_t source = 0; source < 2; ++source) {
        for (size_t target = 0; target < 2; ++target) {
          gsl::at(temp, target) =
              second_jac.get(source, 0) * first_jac.get(0, target);
          for (size_t dummy = 1; dummy < 2; ++dummy) {
            gsl::at(temp, target) +=
                second_jac.get(source, dummy) * first_jac.get(dummy, target);
          }
        }
        for (size_t target = 0; target < 2; ++target) {
          second_jac.get(source, target) = gsl::at(temp, target);
        }
      }
      return second_jac;
    }(i);

    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        INFO(i);
        INFO(j);
        INFO(k);
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = double_rotated2d.inv_jacobian(
        Point<2, Frame::Logical>{{{coord[0], coord[1]}}});

    const auto expected_inv_jac = [&first_rotated2d, &second_rotated2d,
                                   &coords2d](const size_t ii) {
      auto first_inv_jac = first_rotated2d.inv_jacobian(gsl::at(coords2d, ii));

      const auto second_inv_jac =
          second_rotated2d.inv_jacobian(first_rotated2d(gsl::at(coords2d, ii)));

      std::array<double, 2> temp{};
      for (size_t source = 0; source < 2; ++source) {
        for (size_t target = 0; target < 2; ++target) {
          gsl::at(temp, target) =
              first_inv_jac.get(source, 0) * second_inv_jac.get(0, target);
          for (size_t dummy = 1; dummy < 2; ++dummy) {
            gsl::at(temp, target) += first_inv_jac.get(source, dummy) *
                                     second_inv_jac.get(dummy, target);
          }
        }
        for (size_t target = 0; target < 2; ++target) {
          first_inv_jac.get(source, target) = gsl::at(temp, target);
        }
      }

      return first_inv_jac;
    }(i);

    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        INFO(i);
        INFO(j);
        INFO(k);
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }

  // Test 3D
  const auto double_rotated3d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          rotate3d{M_PI_4, M_PI_4, M_PI_2}, rotate3d{M_PI_2, M_PI_4, M_PI_4});
  const auto first_rotated3d = rotate3d{M_PI_4, M_PI_4, M_PI_2};
  const auto second_rotated3d = rotate3d{M_PI_2, M_PI_4, M_PI_4};

  std::array<std::array<double, 3>, 4> coords3d{{{{0.1, 2.8, 9.3}},
                                                 {{-8.2, 2.8, -9.7}},
                                                 {{5.7, -4.9, 8.1}},
                                                 {{2.9, 3.4, -7.8}}}};

  for (size_t i = 0; i < coords3d.size(); ++i) {
    const auto coord = gsl::at(coords3d, i);
    CHECK((make_array<double, 3>(double_rotated3d(
              Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}}))) ==
          second_rotated3d(first_rotated3d(coord)));
    CHECK((make_array<double, 3>(double_rotated3d.inverse(
              Point<3, Frame::Grid>{{{coord[0], coord[1], coord[2]}}}))) ==
          first_rotated3d.inverse(second_rotated3d.inverse(coord)));

    const auto jac = double_rotated3d.jacobian(
        Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});

    const auto expected_jac = [&first_rotated3d, &second_rotated3d,
                               &coords3d](const size_t ii) {

      const auto first_jac = first_rotated3d.jacobian(gsl::at(coords3d, ii));
      auto second_jac =
          second_rotated3d.jacobian(first_rotated3d(gsl::at(coords3d, ii)));

      std::array<double, 3> temp{};
      for (size_t source = 0; source < 3; ++source) {
        for (size_t target = 0; target < 3; ++target) {
          gsl::at(temp, target) =
              second_jac.get(source, 0) * first_jac.get(0, target);
          for (size_t dummy = 1; dummy < 3; ++dummy) {
            gsl::at(temp, target) +=
                second_jac.get(source, dummy) * first_jac.get(dummy, target);
          }
        }
        for (size_t target = 0; target < 3; ++target) {
          second_jac.get(source, target) = gsl::at(temp, target);
        }
      }
      return second_jac;
    }(i);

    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        INFO(i);
        INFO(j);
        INFO(k);
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = double_rotated3d.inv_jacobian(
        Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});

    const auto expected_inv_jac = [&first_rotated3d, &second_rotated3d,
                                   &coords3d](const size_t ii) {
      auto first_inv_jac = first_rotated3d.inv_jacobian(gsl::at(coords3d, ii));

      const auto second_inv_jac =
          second_rotated3d.inv_jacobian(first_rotated3d(gsl::at(coords3d, ii)));

      std::array<double, 3> temp{};
      for (size_t source = 0; source < 3; ++source) {
        for (size_t target = 0; target < 3; ++target) {
          gsl::at(temp, target) =
              first_inv_jac.get(source, 0) * second_inv_jac.get(0, target);
          for (size_t dummy = 1; dummy < 3; ++dummy) {
            gsl::at(temp, target) += first_inv_jac.get(source, dummy) *
                                     second_inv_jac.get(dummy, target);
          }
        }
        for (size_t target = 0; target < 3; ++target) {
          first_inv_jac.get(source, target) = gsl::at(temp, target);
        }
      }

      return first_inv_jac;
    }(i);

    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        INFO(i);
        INFO(j);
        INFO(k);
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }

  // Check inequivalence operator
  CHECK_FALSE(double_rotated3d != double_rotated3d);
  CHECK(double_rotated3d == serialize_and_deserialize(double_rotated3d));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMap", "[Domain][Unit]") {
  test_single_coordinate_map();
  test_coordinate_map_with_affine_map();
  test_coordinate_map_with_rotation_map();
}
