// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ConcentricShell.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
void test_concentric_shell_radial_extent() {
  const auto shell_mesh = [](const Spectral::Quadrature radial_quadrature) {
    return Mesh<3>{
        {{6, 8, 15}},
        {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
          Spectral::Basis::SphericalHarmonic}},
        {{radial_quadrature, Spectral::Quadrature::Gauss,
          Spectral::Quadrature::Equiangular}}};
  };
  const auto shell_radii = [](const Mesh<3>& mesh,
                              const std::array<double, 3>& center) {
    const auto logical = logical_coordinates(mesh);
    const DataVector rho =
        1.0 / (0.5 * (1.0 / 1.5 + 1.0 / 2.5) +
               0.5 * (1.0 / 2.5 - 1.0 / 1.5) * get<0>(logical));
    const DataVector& theta = get<1>(logical);
    const DataVector& phi = get<2>(logical);
    const DataVector x0 = rho * sin(theta) * cos(phi);
    const DataVector y0 = rho * sin(theta) * sin(phi);
    const DataVector z0 = rho * cos(theta);
    const double angle = 0.7;
    const DataVector x = cos(angle) * x0 - sin(angle) * z0;
    const DataVector z = sin(angle) * x0 + cos(angle) * z0;
    return DataVector{sqrt(square(x - center[0]) + square(y0 - center[1]) +
                           square(z - center[2]))};
  };

  const Mesh<3> mesh = shell_mesh(Spectral::Quadrature::GaussLobatto);
  const auto extent = domain::concentric_shell_radial_extent(
      mesh, shell_radii(mesh, {{0.0, 0.0, 0.0}}));
  REQUIRE(extent.has_value());
  // The extent is padded by a small tolerance for roundoff
  CHECK(extent->first <= 1.5);
  CHECK(extent->first > 1.5 - 1.0e-10);
  CHECK(extent->second >= 2.5);
  CHECK(extent->second < 2.5 + 1.0e-10);

  CHECK_FALSE(domain::concentric_shell_radial_extent(
                  mesh, shell_radii(mesh, {{1.0e-8, 0.0, 0.0}}))
                  .has_value());
  const Mesh<3> gauss_mesh = shell_mesh(Spectral::Quadrature::Gauss);
  CHECK_FALSE(domain::concentric_shell_radial_extent(
                  gauss_mesh, shell_radii(gauss_mesh, {{0.0, 0.0, 0.0}}))
                  .has_value());
  const Mesh<3> cube_mesh{6, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  CHECK_FALSE(
      domain::concentric_shell_radial_extent(
          cube_mesh, DataVector{2.0 + get<0>(logical_coordinates(cube_mesh))})
          .has_value());
}

#ifdef SPECTRE_DEBUG
void test_errors() {
  CHECK_THROWS_WITH(domain::concentric_shell_radial_extent(
                        Mesh<3>{4, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto},
                        DataVector{3, 1.0}),
                    Catch::Matchers::ContainsSubstring(
                        "The mesh has 64 grid points but there are 3 radii."));
}
#endif

SPECTRE_TEST_CASE("Unit.Domain.ConcentricShell", "[Domain][Unit]") {
  test_concentric_shell_radial_extent();
#ifdef SPECTRE_DEBUG
  test_errors();
#endif
}
}  // namespace
