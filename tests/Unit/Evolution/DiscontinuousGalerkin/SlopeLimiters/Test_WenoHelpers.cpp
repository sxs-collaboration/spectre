// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {

void test_oscillation_indicator_1d() noexcept {
  const Mesh<1> mesh(5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const DataVector& x = get<0>(logical_coords);

  const auto data = DataVector{1. + x - pow<4>(x)};
  const auto indicator =
      SlopeLimiters::Weno_detail::oscillation_indicator(data, mesh);

  // Expected result computed in Mathematica:
  // f[x_] := 1 + x - x^4
  // Integrate[Sum[Evaluate[D[f[x], {x, i}]^2], {i, 1, 4}], {x, -1, 1}]
  const double expected = 56006. / 35.;
  CHECK(indicator == approx(expected));
}

void test_oscillation_indicator_2d() noexcept {
  const Mesh<2> mesh({{4, 5}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const DataVector& x = get<0>(logical_coords);
  const DataVector& y = get<1>(logical_coords);

  const auto data =
      DataVector{square(x) + cube(y) - 2.5 * x * y + square(x) * y};
  const auto indicator =
      SlopeLimiters::Weno_detail::oscillation_indicator(data, mesh);

  // Expected result computed in Mathematica:
  // g[x_, y_] := x^2 + y^3 - (5/2) x y + x^2 y
  // Integrate[
  //  Sum[Evaluate[D[g[x, y], {x, i}, {y, j}]^2], {i, 1, 3}, {j, 1, 4}]
  //   + Sum[Evaluate[D[g[x, y], {x, i}]^2], {i, 1, 3}]
  //   + Sum[Evaluate[D[g[x, y], {y, j}]^2], {j, 1, 4}],
  //  {x, -1, 1}, {y, -1, 1}]
  const double expected = 2647. / 9.;
  CHECK(indicator == approx(expected));
}

void test_oscillation_indicator_3d() noexcept {
  const Mesh<3> mesh({{4, 3, 5}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const DataVector& x = get<0>(logical_coords);
  const DataVector& y = get<1>(logical_coords);
  const DataVector& z = get<2>(logical_coords);

  const auto data = DataVector{square(x) + 2. * y + z - 6. * cube(z) -
                               3. * x * square(y) * cube(z) - x * y + y * z};
  const auto indicator =
      SlopeLimiters::Weno_detail::oscillation_indicator(data, mesh);

  // Expected result computed in Mathematica:
  // h[x_, y_, z_] := x^2 + 2 y + z - 6 z^3 - 3 x y^2 z^3 - x y + y z
  // Integrate[
  //  Sum[Evaluate[D[h[x, y, z], {x, i}, {y, j}, {z, k}]^2],
  //      {i, 1, 3}, {j, 1, 2}, {k, 1, 4}]
  //   + Sum[Evaluate[ D[h[x, y, z], {x, i}, {y, j}]^2], {i, 1, 3}, {j, 1, 2}]
  //   + Sum[Evaluate[ D[h[x, y, z], {x, i}, {z, k}]^2], {i, 1, 3}, {k, 1, 4}]
  //   + Sum[Evaluate[ D[h[x, y, z], {y, j}, {z, k}]^2], {j, 1, 2}, {k, 1, 4}]
  //   + Sum[Evaluate[ D[h[x, y, z], {x, i}]^2], {i, 1, 3}]
  //   + Sum[Evaluate[ D[h[x, y, z], {y, j}]^2], {j, 1, 2}]
  //   + Sum[Evaluate[ D[h[x, y, z], {z, k}]^2], {k, 1, 4}],
  //  {x, -1, 1}, {y, -1, 1}, {z, -1, 1}]
  const double expected = 3066352. / 75.;
  CHECK(indicator == approx(expected));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.WenoHelpers",
                  "[SlopeLimiters][Unit]") {
  test_oscillation_indicator_1d();
  test_oscillation_indicator_2d();
  test_oscillation_indicator_3d();
}
