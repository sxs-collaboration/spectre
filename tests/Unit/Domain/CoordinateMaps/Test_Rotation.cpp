// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare domain::CoordinateMaps::Rotation

namespace domain {
namespace {
template <typename T>
void test_rotation_3(const CoordinateMaps::Rotation<3>& three_dim_rotation_map,
                     const std::array<T, 3>& xi_hat,
                     const std::array<T, 3>& eta_hat,
                     const std::array<T, 3>& zeta_hat) {
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
    CHECK(gsl::at(three_dim_rotation_map.inverse(zero_grid).get(), i) ==
          approx(gsl::at(zero_logical, i)));
    CHECK(gsl::at(three_dim_rotation_map.inverse(xi_hat).get(), i) ==
          approx(gsl::at(xi, i)));
    CHECK(gsl::at(three_dim_rotation_map.inverse(eta_hat).get(), i) ==
          approx(gsl::at(eta, i)));
    CHECK(gsl::at(three_dim_rotation_map.inverse(zeta_hat).get(), i) ==
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
  test_serialization(three_dim_rotation_map);

  test_coordinate_map_argument_types(three_dim_rotation_map, xi);
}

void test_rotation_2() {
  INFO("Rotation 2");
  CoordinateMaps::Rotation<2> half_pi_rotation_map(M_PI_2);

  const auto xi0 = make_array<2>(0.0);
  const auto x0 = make_array<2>(0.0);

  CHECK(half_pi_rotation_map(xi0)[0] == approx(x0[0]));
  CHECK(half_pi_rotation_map(xi0)[1] == approx(x0[1]));

  CHECK(half_pi_rotation_map.inverse(x0).get()[0] == approx(xi0[0]));
  CHECK(half_pi_rotation_map.inverse(x0).get()[1] == approx(xi0[1]));

  const std::array<double, 2> xi1{{1.0, 0.0}};
  const std::array<double, 2> x1{{0.0, 1.0}};

  CHECK(half_pi_rotation_map(xi1)[0] == approx(x1[0]));
  CHECK(half_pi_rotation_map(xi1)[1] == approx(x1[1]));

  CHECK(half_pi_rotation_map.inverse(x1).get()[0] == approx(xi1[0]));
  CHECK(half_pi_rotation_map.inverse(x1).get()[1] == approx(xi1[1]));

  const std::array<double, 2> xi2{{0.0, 1.0}};
  const std::array<double, 2> x2{{-1.0, 0.0}};

  CHECK(half_pi_rotation_map(xi2)[0] == approx(x2[0]));
  CHECK(half_pi_rotation_map(xi2)[1] == approx(x2[1]));

  CHECK(half_pi_rotation_map.inverse(x2).get()[0] == approx(xi2[0]));
  CHECK(half_pi_rotation_map.inverse(x2).get()[1] == approx(xi2[1]));

  const auto inv_jac = half_pi_rotation_map.inv_jacobian(xi2);
  CHECK((get<0, 0>(inv_jac)) == approx(0.0));
  CHECK((get<0, 1>(inv_jac)) == approx(1.0));
  CHECK((get<1, 0>(inv_jac)) == approx(-1.0));
  CHECK((get<1, 1>(inv_jac)) == approx(0.0));

  const auto jac = half_pi_rotation_map.jacobian(xi2);
  CHECK((get<0, 0>(jac)) == approx(0.0));
  CHECK((get<0, 1>(jac)) == approx(-1.0));
  CHECK((get<1, 0>(jac)) == approx(1.0));
  CHECK((get<1, 1>(jac)) == approx(0.0));

  // Check inequivalence operator
  CHECK_FALSE(half_pi_rotation_map != half_pi_rotation_map);
  test_serialization(half_pi_rotation_map);

  test_coordinate_map_argument_types(half_pi_rotation_map, xi1);
  check_if_map_is_identity(CoordinateMaps::Rotation<2>{0.0});
  CHECK(not CoordinateMaps::Rotation<2>(M_PI_2).is_identity());
}

void test_rotation_3() {
  INFO("Rotation 3");
  test_rotation_3(CoordinateMaps::Rotation<3>(0.0, 0.0, 0.0),
                  std::array<double, 3>{{1.0, 0.0, 0.0}},
                  std::array<double, 3>{{0.0, 1.0, 0.0}},
                  std::array<double, 3>{{0.0, 0.0, 1.0}});

  test_rotation_3(CoordinateMaps::Rotation<3>(M_PI_2, 0.0, 0.0),
                  std::array<double, 3>{{0.0, 1.0, 0.0}},
                  std::array<double, 3>{{-1.0, 0.0, 0.0}},
                  std::array<double, 3>{{0.0, 0.0, 1.0}});

  test_rotation_3(CoordinateMaps::Rotation<3>(M_PI, 0.0, 0.0),
                  std::array<double, 3>{{-1.0, 0.0, 0.0}},
                  std::array<double, 3>{{0.0, -1.0, 0.0}},
                  std::array<double, 3>{{0.0, 0.0, 1.0}});

  test_rotation_3(CoordinateMaps::Rotation<3>(-M_PI_2, 0.0, 0.0),
                  std::array<double, 3>{{0.0, -1.0, 0.0}},
                  std::array<double, 3>{{1.0, 0.0, 0.0}},
                  std::array<double, 3>{{0.0, 0.0, 1.0}});

  test_rotation_3(CoordinateMaps::Rotation<3>(0.0, -M_PI_2, 0.0),
                  std::array<double, 3>{{0.0, 0.0, 1.0}},
                  std::array<double, 3>{{0.0, 1.0, 0.0}},
                  std::array<double, 3>{{-1.0, 0.0, 0.0}});

  test_rotation_3(CoordinateMaps::Rotation<3>(0.0, M_PI_2, 0.0),
                  std::array<double, 3>{{0.0, 0.0, -1.0}},
                  std::array<double, 3>{{0.0, 1.0, 0.0}},
                  std::array<double, 3>{{1.0, 0.0, 0.0}});
  check_if_map_is_identity(CoordinateMaps::Rotation<3>{0.0, 0.0, 0.0});
  CHECK(not CoordinateMaps::Rotation<3>{0.0, 0.0, M_PI_2}.is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Rotation", "[Domain][Unit]") {
  test_rotation_2();
  test_rotation_3();
}
}  // namespace domain
