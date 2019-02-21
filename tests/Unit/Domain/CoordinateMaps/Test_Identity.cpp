// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "Domain/CoordinateMaps/Identity.hpp"
#include "Utilities/MakeArray.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test_identity() {
  CoordinateMaps::Identity<Dim> identity_map;
  const auto xi = make_array<Dim>(1.0);
  const auto x = make_array<Dim>(1.0);
  CHECK(identity_map(xi) == x);
  CHECK(identity_map.inverse(x).get() == xi);
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
  test_serialization(identity_map);

  test_coordinate_map_argument_types(identity_map, xi);

  check_if_map_is_identity(CoordinateMaps::Identity<Dim>{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Identity", "[Domain][Unit]") {
  test_identity<1>();
  test_identity<2>();
  test_identity<3>();
}
}  // namespace domain
