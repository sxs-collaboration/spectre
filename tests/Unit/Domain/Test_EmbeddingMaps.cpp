// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/EmbeddingMaps/AffineMap.hpp"
#include "Domain/EmbeddingMaps/Identity.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

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
