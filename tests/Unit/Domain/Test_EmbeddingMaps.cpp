// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/EmbeddingMaps/Identity.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

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
