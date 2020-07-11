// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"

namespace {

void test_derivative_weight() noexcept {
  INFO("Test DerivativeWeight");

  CHECK(get_output(Limiters::Weno_detail::DerivativeWeight::Unity) == "Unity");
  CHECK(get_output(Limiters::Weno_detail::DerivativeWeight::PowTwoEll) ==
        "PowTwoEll");
  CHECK(
      get_output(
          Limiters::Weno_detail::DerivativeWeight::PowTwoEllOverEllFactorial) ==
      "PowTwoEllOverEllFactorial");
}

void test_oscillation_indicator_1d() noexcept {
  INFO("Test oscillation_indicator in 1D");
  const Mesh<1> mesh(5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const DataVector& x = get<0>(logical_coords);

  const auto data = DataVector{1. + x - pow<4>(x)};
  const auto indicator = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::Unity, data, mesh);

  // Expected result computed in Mathematica:
  // f[x_] := 1 + x - x^4
  // w[i_] := 1
  // Integrate[Sum[w[i] Evaluate[D[f[x], {x, i}]^2], {i, 1, 4}], {x, -1, 1}]
  const double expected = 56006. / 35.;
  CHECK(indicator == approx(expected));

  // As above, but with derivative weights given by
  // w[i_] := 2^(2 i - 1)
  const auto indicator2 = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::PowTwoEll, data, mesh);
  const double expected2 = 5607628. / 35.;
  CHECK(indicator2 == approx(expected2));

  // Again as above, but with derivative weights given by
  // w[i_] := 2^(2 i - 1) / (i!)^2
  const auto indicator3 = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::PowTwoEllOverEllFactorial, data,
      mesh);
  const double expected3 = 76196. / 105.;
  CHECK(indicator3 == approx(expected3));
}

void test_oscillation_indicator_2d() noexcept {
  INFO("Test oscillation_indicator in 2D");
  const Mesh<2> mesh({{4, 5}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const DataVector& x = get<0>(logical_coords);
  const DataVector& y = get<1>(logical_coords);

  const auto data =
      DataVector{square(x) + cube(y) - 2.5 * x * y + square(x) * y};
  const auto indicator = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::Unity, data, mesh);

  // Expected result computed in Mathematica:
  // g[x_, y_] := x^2 + y^3 - (5/2) x y + x^2 y
  // w[i_] := 1
  // Integrate[
  //  Sum[w[i] w[j] Evaluate[D[g[x, y], {x, i}, {y, j}]^2],
  //      {i, 1, 3}, {j, 1, 4}]
  //   + Sum[w[i] w[0] Evaluate[D[g[x, y], {x, i}]^2], {i, 1, 3}]
  //   + Sum[w[0] w[j] Evaluate[D[g[x, y], {y, j}]^2], {j, 1, 4}],
  //  {x, -1, 1}, {y, -1, 1}]
  const double expected = 2647. / 9.;
  CHECK(indicator == approx(expected));

  // w[i_] := 2^(2 i - 1)
  const auto indicator2 = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::PowTwoEll, data, mesh);
  const double expected2 = 26938. / 9.;
  CHECK(indicator2 == approx(expected2));

  // w[i_] := 2^(2 i - 1) / (i!)^2
  const auto indicator3 = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::PowTwoEllOverEllFactorial, data,
      mesh);
  const double expected3 = 3178. / 9.;
  CHECK(indicator3 == approx(expected3));
}

void test_oscillation_indicator_3d() noexcept {
  INFO("Test oscillation_indicator in 3D");
  const Mesh<3> mesh({{4, 3, 5}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const DataVector& x = get<0>(logical_coords);
  const DataVector& y = get<1>(logical_coords);
  const DataVector& z = get<2>(logical_coords);

  const auto data = DataVector{square(x) + 2. * y + z - 6. * cube(z) -
                               3. * x * square(y) * cube(z) - x * y + y * z};
  const auto indicator = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::Unity, data, mesh);

  // Expected result computed in Mathematica:
  // h[x_, y_, z_] := x^2 + 2 y + z - 6 z^3 - 3 x y^2 z^3 - x y + y z
  // w[i_] := 1
  // Integrate[
  //  Sum[w[i] w[j] w[k] Evaluate[D[h[x, y, z], {x, i}, {y, j}, {z, k}]^2],
  //      {i, 1, 3}, {j, 1, 2}, {k, 1, 4}]
  //   + Sum[w[i] w[j] w[0] Evaluate[ D[h[x, y, z], {x, i}, {y, j}]^2],
  //         {i, 1, 3}, {j, 1, 2}]
  //   + Sum[w[i] w[0] w[k] Evaluate[ D[h[x, y, z], {x, i}, {z, k}]^2],
  //         {i, 1, 3}, {k, 1, 4}]
  //   + Sum[w[0] w[j] w[k] Evaluate[ D[h[x, y, z], {y, j}, {z, k}]^2],
  //         {j, 1, 2}, {k, 1, 4}]
  //   + Sum[w[i] w[0] w[0] Evaluate[ D[h[x, y, z], {x, i}]^2], {i, 1, 3}]
  //   + Sum[w[0] w[j] w[0] Evaluate[ D[h[x, y, z], {y, j}]^2], {j, 1, 2}]
  //   + Sum[w[0] w[0] w[k] Evaluate[ D[h[x, y, z], {z, k}]^2], {k, 1, 4}],
  //  {x, -1, 1}, {y, -1, 1}, {z, -1, 1}]
  const double expected = 3066352. / 75.;
  CHECK(indicator == approx(expected));

  // w[i_] := 2^(2 i - 1)
  const auto indicator2 = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::PowTwoEll, data, mesh);
  const double expected2 = 3611348444. / 525.;
  CHECK(indicator2 == approx(expected2));

  // w[i_] := 2^(2 i - 1) / (i!)^2
  const auto indicator3 = Limiters::Weno_detail::oscillation_indicator(
      Limiters::Weno_detail::DerivativeWeight::PowTwoEllOverEllFactorial, data,
      mesh);
  const double expected3 = 54886604. / 525.;
  CHECK(indicator3 == approx(expected3));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.Weno.OscillationIndicator",
                  "[Limiters][Unit]") {
  test_derivative_weight();

  test_oscillation_indicator_1d();
  test_oscillation_indicator_2d();
  test_oscillation_indicator_3d();
}
