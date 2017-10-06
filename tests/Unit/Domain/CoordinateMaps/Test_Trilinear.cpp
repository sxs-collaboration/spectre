// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Trilinear.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Trilinear", "[Domain][Unit]") {
  // Set up random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);

  const std::array<double, 3> x0{{0, 0, 0}};
  const std::array<double, 3> x1{{1, 0, 0}};
  const std::array<double, 3> x2{{0, 1, 0}};
  const std::array<double, 3> x3{{1, 1, 0}};
  const std::array<double, 3> x4{{0, 0, 1}};
  const std::array<double, 3> x5{{1, 0, 1}};
  const std::array<double, 3> x6{{0, 1, 1}};
  const std::array<double, 3> x7{{1, 1, 1}};

  CoordinateMaps::Trilinear trilinear(
      std::array<std::array<double, 3>, 8>{{x0, x1, x2, x3, x4, x5, x6, x7}});

  const std::array<double, 3> xi0{{-1, -1, -1}};
  const std::array<double, 3> xi1{{1, -1, -1}};
  const std::array<double, 3> xi2{{-1, 1, -1}};
  const std::array<double, 3> xi3{{1, 1, -1}};
  const std::array<double, 3> xi4{{-1, -1, 1}};
  const std::array<double, 3> xi5{{1, -1, 1}};
  const std::array<double, 3> xi6{{-1, 1, 1}};
  const std::array<double, 3> xi7{{1, 1, 1}};

  CHECK(trilinear(xi0) == x0);
  CHECK(trilinear(xi1) == x1);
  CHECK(trilinear(xi2) == x2);

  const double inv_jacobian_00 = 2;

  CHECK((trilinear.inv_jacobian(xi0).template get<0, 0>()) == inv_jacobian_00);
  CHECK((trilinear.inv_jacobian(xi1).template get<0, 0>()) == inv_jacobian_00);
  CHECK((trilinear.inv_jacobian(xi2).template get<0, 0>()) == inv_jacobian_00);

  const double jacobian_00 = 0.5;
  CHECK((trilinear.jacobian(xi0).template get<0, 0>()) == jacobian_00);
  CHECK((trilinear.jacobian(xi1).template get<0, 0>()) == jacobian_00);
  CHECK((trilinear.jacobian(xi2).template get<0, 0>()) == jacobian_00);

  // Check inequivalence operator
  CHECK_FALSE(trilinear != trilinear);
  CHECK(trilinear == serialize_and_deserialize(trilinear));

  const double dxi = 1e-5;
  const double deta = 1e-5;
  const double xi = real_dis(gen);
  CAPTURE_PRECISE(xi);
  const double eta = real_dis(gen);
  CAPTURE_PRECISE(eta);
  const double zeta = real_dis(gen);
  CAPTURE_PRECISE(zeta);
  const double r00 = real_dis(gen);
  CAPTURE_PRECISE(r00);
  const double r01 = real_dis(gen);
  CAPTURE_PRECISE(r01);
  const double r02 = real_dis(gen);
  CAPTURE_PRECISE(r02);
  const double r10 = real_dis(gen);
  CAPTURE_PRECISE(r10);
  const double r11 = real_dis(gen);
  CAPTURE_PRECISE(r11);
  const double r12 = real_dis(gen);
  CAPTURE_PRECISE(r12);
  const double r20 = real_dis(gen);
  CAPTURE_PRECISE(r20);
  const double r21 = real_dis(gen);
  CAPTURE_PRECISE(r21);
  const double r22 = real_dis(gen);
  CAPTURE_PRECISE(r22);
  const double r30 = real_dis(gen);
  CAPTURE_PRECISE(r30);
  const double r31 = real_dis(gen);
  CAPTURE_PRECISE(r31);
  const double r32 = real_dis(gen);
  CAPTURE_PRECISE(r32);
  const double r40 = real_dis(gen);
  CAPTURE_PRECISE(r40);
  const double r41 = real_dis(gen);
  CAPTURE_PRECISE(r41);
  const double r42 = real_dis(gen);
  CAPTURE_PRECISE(r42);
  const double r50 = real_dis(gen);
  CAPTURE_PRECISE(r50);
  const double r51 = real_dis(gen);
  CAPTURE_PRECISE(r51);
  const double r52 = real_dis(gen);
  CAPTURE_PRECISE(r52);
  const double r60 = real_dis(gen);
  CAPTURE_PRECISE(r60);
  const double r61 = real_dis(gen);
  CAPTURE_PRECISE(r61);
  const double r62 = real_dis(gen);
  CAPTURE_PRECISE(r62);
  const double r70 = real_dis(gen);
  CAPTURE_PRECISE(r70);
  const double r71 = real_dis(gen);
  CAPTURE_PRECISE(r71);
  const double r72 = real_dis(gen);
  CAPTURE_PRECISE(r72);

  const std::array<double, 3> r0{{r00, r01, r02}};
  const std::array<double, 3> r1{{r10, r11, r12}};
  const std::array<double, 3> r2{{r20, r21, r22}};
  const std::array<double, 3> r3{{r30, r31, r32}};
  const std::array<double, 3> r4{{r40, r41, r42}};
  const std::array<double, 3> r5{{r50, r51, r52}};
  const std::array<double, 3> r6{{r60, r61, r62}};
  const std::array<double, 3> r7{{r70, r71, r72}};

  const CoordinateMaps::Trilinear map(
      std::array<std::array<double, 3>, 8>{{r0, r1, r2, r3, r4, r5, r6, r7}});
  const std::array<double, 3> test_point{{xi, eta, zeta}};

  test_jacobian(map, test_point);
  test_inv_jacobian(map, test_point);

  CHECK(serialize_and_deserialize(map) == map);

  test_coordinate_map_implementation(map);
}
