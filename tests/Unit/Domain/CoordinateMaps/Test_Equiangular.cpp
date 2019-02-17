// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Equiangular", "[Domain][Unit]") {
  const double xA = -1.0;
  const double xB = 2.0;
  const double xa = -3.0;
  const double xb = 2.0;

  CoordinateMaps::Equiangular equiangular_map(xA, xB, xa, xb);

  const double xi = 0.5 * (xA + xB);
  const double x = xb * (xi - xA) / (xB - xA) + xa * (xB - xi) / (xB - xA);
  const double r1 = 0.172899453;
  const double r2 = -0.234256219;

  const std::array<double, 1> point_A{{xA}};
  const std::array<double, 1> point_B{{xB}};
  const std::array<double, 1> point_a{{xa}};
  const std::array<double, 1> point_b{{xb}};
  const std::array<double, 1> point_xi{{xi}};
  const std::array<double, 1> point_x{{x}};
  const std::array<double, 1> point_r1{{r1}};
  const std::array<double, 1> point_r2{{r2}};

  CHECK(equiangular_map(point_A)[0] == approx(point_a[0]));
  CHECK(equiangular_map(point_B)[0] == approx(point_b[0]));
  CHECK(equiangular_map(point_xi)[0] == approx(point_x[0]));
  CHECK(equiangular_map(point_r1)[0] ==
        approx(-0.5 + 2.5 * tan(M_PI_4 * (2.0 * r1 - 1.0) / 3.0)));
  CHECK(equiangular_map(point_r2)[0] ==
        approx(-0.5 + 2.5 * tan(M_PI_4 * (2.0 * r2 - 1.0) / 3.0)));

  CHECK(equiangular_map.inverse(point_a).get()[0] == approx(point_A[0]));
  CHECK(equiangular_map.inverse(point_b).get()[0] == approx(point_B[0]));
  CHECK(equiangular_map.inverse(point_x).get()[0] == approx(point_xi[0]));
  CHECK(equiangular_map.inverse(point_r1).get()[0] ==
        approx(0.5 + 3.0 * atan(0.4 * r1 + 0.2) / M_PI_2));
  CHECK(equiangular_map.inverse(point_r2).get()[0] ==
        approx(0.5 + 3.0 * atan(0.4 * r2 + 0.2) / M_PI_2));

  CHECK((get<0, 0>(equiangular_map.inv_jacobian(point_A))) *
            (get<0, 0>(equiangular_map.jacobian(point_A))) ==
        approx(1.0));
  CHECK((get<0, 0>(equiangular_map.inv_jacobian(point_B))) *
            (get<0, 0>(equiangular_map.jacobian(point_B))) ==
        approx(1.0));
  CHECK((get<0, 0>(equiangular_map.inv_jacobian(point_xi))) *
            (get<0, 0>(equiangular_map.jacobian(point_xi))) ==
        approx(1.0));

  test_jacobian(equiangular_map, point_xi);
  test_inv_jacobian(equiangular_map, point_xi);
  test_inverse_map(equiangular_map, point_xi);

  // Check inequivalence operator
  CHECK_FALSE(equiangular_map != equiangular_map);
  test_serialization(equiangular_map);

  test_coordinate_map_argument_types(equiangular_map, point_xi);
  test_coordinate_map_argument_types(equiangular_map, point_r1);
  test_coordinate_map_argument_types(equiangular_map, point_r2);
  CHECK(not CoordinateMaps::Equiangular{}.is_identity());
}
}  // namespace domain
