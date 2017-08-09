// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/EmbeddingMaps/Identity.hpp"
#include "tests/Unit/TestHelpers.hpp"

template <size_t Dim>
void test_identity() {
  CoordinateMaps::Identity<Dim> identity_map;
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

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Identity", "[Domain][Unit]") {
  test_identity<1>();
  test_identity<2>();
}
