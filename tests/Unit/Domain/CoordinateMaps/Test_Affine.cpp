// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace domain {
SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Affine", "[Domain][Unit]") {
  const double xA = -1.0;
  const double xB = 1.0;
  const double xa = -2.0;
  const double xb = 2.0;

  CoordinateMaps::Affine affine_map(xA, xB, xa, xb);

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

  CHECK(affine_map.inverse(point_a).value() == point_A);
  CHECK(affine_map.inverse(point_b).value() == point_B);
  CHECK(affine_map.inverse(point_x).value() == point_xi);

  const double inv_jacobian_00 = (xB - xA) / (xb - xa);

  CHECK((get<0, 0>(affine_map.inv_jacobian(point_A))) == inv_jacobian_00);
  CHECK((get<0, 0>(affine_map.inv_jacobian(point_B))) == inv_jacobian_00);
  CHECK((get<0, 0>(affine_map.inv_jacobian(point_xi))) == inv_jacobian_00);

  const double jacobian_00 = (xb - xa) / (xB - xA);
  CHECK((get<0, 0>(affine_map.jacobian(point_A))) == jacobian_00);
  CHECK((get<0, 0>(affine_map.jacobian(point_B))) == jacobian_00);
  CHECK((get<0, 0>(affine_map.jacobian(point_xi))) == jacobian_00);

  // Check inequivalence operator
  CHECK_FALSE(affine_map != affine_map);
  test_serialization(affine_map);

  test_coordinate_map_argument_types(affine_map, point_xi);

  check_if_map_is_identity(CoordinateMaps::Affine{-1.0, 1.0, -1.0, 1.0});
}
}  // namespace domain
