// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <random>

#include "Domain/CoordinateMaps/CylindricalSphericalShell.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace domain {
namespace {

void test_cylindrical_spherical_shell() {
  INFO("CylindricalSphericalShell");
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);
  std::uniform_real_distribution<> xi_dis(-1.0, 1.0);
  std::uniform_real_distribution<> angle_dis(-M_PI, M_PI);

  const double r_sphere = 0.5 + 0.5 * unit_dis(gen);
  CAPTURE(r_sphere);
  const double r_inner = r_sphere * (0.1 + 0.5 * unit_dis(gen));
  CAPTURE(r_inner);

  const double x_inner_lower = -r_sphere * (0.2 + 0.3 * unit_dis(gen));
  const double x_inner_upper = r_sphere * (0.2 + 0.3 * unit_dis(gen));
  const double x_outer_lower = -r_sphere * (0.1 + 0.2 * unit_dis(gen));
  const double x_outer_upper = r_sphere * (0.1 + 0.2 * unit_dis(gen));
  CAPTURE(x_inner_lower);
  CAPTURE(x_inner_upper);
  CAPTURE(x_outer_lower);
  CAPTURE(x_outer_upper);
  const CoordinateMaps::CylindricalSphericalShell map(
      x_inner_lower, x_inner_upper, x_outer_lower, x_outer_upper, r_inner,
      r_sphere);

  // Note: test_suite_for_map_on_cylinder is NOT used here because it expects
  // source coordinates in Cartesian cylindrical form (rho*cos(phi),
  // rho*sin(phi), z), whereas CylindricalSphericalShell takes (xi, eta, zeta)
  // where eta is the azimuthal angle directly.  Instead we test explicitly
  // over the proper source domain.
  const auto test_point = [&](const std::array<double, 3>& source_point) {
    test_jacobian(map, source_point);
    test_inv_jacobian(map, source_point);
    test_inverse_map(map, source_point);
    test_coordinate_map_argument_types(map, source_point);
  };

  // Random interior points.
  for (size_t i = 0; i < 5; ++i) {
    test_point({xi_dis(gen), angle_dis(gen), xi_dis(gen)});
  }

  {
    INFO("Boundary: xi = +1 (outer sphere face) and xi = -1 (inner cylinder)");
    test_point({1.0, angle_dis(gen), xi_dis(gen)});
    test_point({-1.0, angle_dis(gen), xi_dis(gen)});
  }
  {
    INFO("Boundary: zeta = +1 and zeta = -1 (axial faces)");
    test_point({xi_dis(gen), angle_dis(gen), 1.0});
    test_point({xi_dis(gen), angle_dis(gen), -1.0});
  }
  {
    INFO("Corners: (xi, zeta) = (±1, ±1)");
    test_point({1.0, angle_dis(gen), 1.0});
    test_point({1.0, angle_dis(gen), -1.0});
    test_point({-1.0, angle_dis(gen), 1.0});
    test_point({-1.0, angle_dis(gen), -1.0});
  }
  {
    INFO("Full azimuthal range including near ±pi");
    test_point({xi_dis(gen), M_PI - 1.0e-6, xi_dis(gen)});
    test_point({xi_dis(gen), -M_PI + 1.0e-6, xi_dis(gen)});
  }

  // Serialisation and operator==.
  test_serialization(map);
  CHECK_FALSE(map != map);

  // Geometric check: at xi=+1, all images lie on the sphere r^2 + x^2 =
  // r_sphere^2 (centred at the origin).
  const Approx local_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  for (size_t i = 0; i < 5; ++i) {
    const double phi = angle_dis(gen);
    const double zeta = xi_dis(gen);
    const auto image = map(std::array{1.0, phi, zeta});
    const double r_image = std::sqrt(square(image[1]) + square(image[2]));
    CHECK(square(image[0]) + square(r_image) == local_approx(square(r_sphere)));
  }

  // Geometric check: at xi=-1, all images lie on the inner cylinder
  // r = r_inner.
  for (size_t i = 0; i < 5; ++i) {
    const double phi = angle_dis(gen);
    const double zeta = xi_dis(gen);
    const auto image = map(std::array{-1.0, phi, zeta});
    const double r_image = std::sqrt(square(image[1]) + square(image[2]));
    CHECK(r_image == local_approx(r_inner));
  }

  // Geometric check: x-coordinate at zeta=-1 and zeta=+1 blends linearly
  // between x_inner and x_outer as xi varies.
  {
    // At zeta = -1 (beta=0): x should go from x_inner_lower (xi=-1) to
    // x_outer_lower (xi=+1).
    const double phi = angle_dis(gen);
    const auto lo_inner = map(std::array{-1.0, phi, -1.0});
    const auto lo_outer = map(std::array{1.0, phi, -1.0});
    CHECK(lo_inner[0] == local_approx(x_inner_lower));
    CHECK(lo_outer[0] == local_approx(x_outer_lower));
    // At zeta = +1 (beta=1): x_inner_upper at xi=-1, x_outer_upper at xi=+1.
    const auto hi_inner = map(std::array{-1.0, phi, 1.0});
    const auto hi_outer = map(std::array{1.0, phi, 1.0});
    CHECK(hi_inner[0] == local_approx(x_inner_upper));
    CHECK(hi_outer[0] == local_approx(x_outer_upper));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.CylindricalSphericalShell",
                  "[Domain][Unit]") {
  test_cylindrical_spherical_shell();
  CHECK(not CoordinateMaps::CylindricalSphericalShell{}.is_identity());
}
}  // namespace domain
