// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Variables

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

struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Scalar"; }
};

template <size_t VolumeDim>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, VolumeDim>;
  static std::string name() noexcept { return "Vector"; }
};

void test_reconstruction_1d() noexcept {
  const double neighbor_linear_weight = 0.005;
  const Mesh<1> mesh(5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto coords = logical_coordinates(mesh);

  const auto evaluate_polynomial = [&coords](
      const std::array<double, 5>& coeffs) noexcept {
    const auto& x = get<0>(coords);
    return DataVector{coeffs[0] + coeffs[1] * x + coeffs[2] * square(x) +
                      coeffs[3] * cube(x) + coeffs[4] * pow<4>(x)};
  };

  auto scalar =
      ScalarTag::type{{{evaluate_polynomial({{1., 2., 0., 0.5, 0.1}})}}};

  std::unordered_map<std::pair<Direction<1>, ElementId<1>>,
                     Variables<tmpl::list<ScalarTag>>,
                     boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_vars{};
  auto& vars_lower_xi =
      neighbor_vars[std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1))];
  auto& vars_upper_xi =
      neighbor_vars[std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2))];
  vars_lower_xi.initialize(mesh.number_of_grid_points());
  vars_upper_xi.initialize(mesh.number_of_grid_points());
  get<ScalarTag>(vars_lower_xi) =
      ScalarTag::type{{{evaluate_polynomial({{0., 1., 0., 1., 0.}})}}};
  get<ScalarTag>(vars_upper_xi) =
      ScalarTag::type{{{evaluate_polynomial({{0., 0., 1., 1., 2.}})}}};

  SlopeLimiters::Weno_detail::reconstruct_from_weighted_sum<ScalarTag>(
      make_not_null(&scalar), mesh, neighbor_linear_weight, neighbor_vars);

  // Expected result computed in Mathematica by computing oscillation indicator
  // as in oscillation_indicator tests, then WENO weights, then superposition.
  auto expected_reconstructed_scalar = ScalarTag::type{{{evaluate_polynomial(
      {{1.0000250662809542, 1.9987344134217362, 3.25819395292328e-7,
        0.5006326303794342, 0.09987412556290375}})}}};
  CHECK_ITERABLE_APPROX(scalar, expected_reconstructed_scalar);
}

void test_reconstruction_2d() noexcept {
  const double neighbor_linear_weight = 0.001;
  const Mesh<2> mesh({{3, 3}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto coords = logical_coordinates(mesh);

  const auto evaluate_polynomial = [&coords](
      const std::array<double, 9>& coeffs) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    return DataVector{coeffs[0] + coeffs[1] * x + coeffs[2] * square(x) +
                      y * (coeffs[3] + coeffs[4] * x + coeffs[5] * square(x)) +
                      square(y) *
                          (coeffs[6] + coeffs[7] * x + coeffs[8] * square(x))};
  };

  auto vector = VectorTag<2>::type{
      {{evaluate_polynomial({{2., 1., 0., 1.5, 1., 0., 1., 0., 0.}}),
        evaluate_polynomial({{0.5, 1., 0.5, 1.5, 2., 1., 1., 2., 3.}})}}};

  std::unordered_map<std::pair<Direction<2>, ElementId<2>>,
                     Variables<tmpl::list<VectorTag<2>>>,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_vars{};
  auto& vars_lower_xi =
      neighbor_vars[std::make_pair(Direction<2>::lower_xi(), ElementId<2>(1))];
  auto& vars_upper_xi =
      neighbor_vars[std::make_pair(Direction<2>::upper_xi(), ElementId<2>(2))];
  auto& vars_lower_eta =
      neighbor_vars[std::make_pair(Direction<2>::lower_eta(), ElementId<2>(3))];
  auto& vars_upper_eta =
      neighbor_vars[std::make_pair(Direction<2>::upper_eta(), ElementId<2>(4))];
  vars_lower_xi.initialize(mesh.number_of_grid_points());
  vars_upper_xi.initialize(mesh.number_of_grid_points());
  vars_lower_eta.initialize(mesh.number_of_grid_points());
  vars_upper_eta.initialize(mesh.number_of_grid_points());
  get<VectorTag<2>>(vars_lower_xi) = VectorTag<2>::type{
      {{evaluate_polynomial({{0., 1., 0., 0., 1., 1., 0., 0., 0}}),
        evaluate_polynomial({{1., 0.1, 0., 0.1, 0., 0., 0., 0.}})}}};
  get<VectorTag<2>>(vars_upper_xi) = VectorTag<2>::type{
      {{evaluate_polynomial({{0., 0., 1., 1., 2., 1., 0., 1., 1.}}),
        evaluate_polynomial({{0., 1., 0., 1., 0., 0., 0., 0., 0.}})}}};
  get<VectorTag<2>>(vars_lower_eta) = VectorTag<2>::type{
      {{evaluate_polynomial({{1., 0., 0., 0., 0.5, 0., 0., 0., 0.5}}),
        evaluate_polynomial({{1., 0., 1., 1., 0., 0., 0., 0.}})}}};
  get<VectorTag<2>>(vars_upper_eta) = VectorTag<2>::type{
      {{evaluate_polynomial({{1., 0., 0., 0.5, 1., 0., 0., 0., 0.}}),
        evaluate_polynomial({{0., 0., 0., 1., 0., 0., 1., 1., 1.}})}}};

  SlopeLimiters::Weno_detail::reconstruct_from_weighted_sum<VectorTag<2>>(
      make_not_null(&vector), mesh, neighbor_linear_weight, neighbor_vars);

  // Expected result computed in Mathematica by computing oscillation indicator
  // as in oscillation_indicator tests, then WENO weights, then superposition.
  auto expected_reconstructed_vector = VectorTag<2>::type{
      {{evaluate_polynomial(
            {{2.010056442214612, 0.9705606381771584, 0.000026246579961852654,
              1.4682459390241314, 0.9992393252122325, 0.0010535139634810797,
              0.9695333707936392, 0.000026246579961852654,
              0.0008131679476911193}}),
        evaluate_polynomial(
            {{1.3333271694137983, 0.10009196057382466, 0.00001162876091830289,
              0.10010376454909513, 6.629378550236318e-6, 3.314689275118159e-6,
              3.4899036272802205e-6, 6.80459290239838e-6,
              0.00001011928217751654}})}}};
  CHECK_ITERABLE_APPROX(vector, expected_reconstructed_vector);
}

void test_reconstruction_3d() noexcept {
  const double neighbor_linear_weight = 0.001;
  const Mesh<3> mesh({{3, 3, 3}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto coords = logical_coordinates(mesh);

  // 3D case has so many modes... so we simplify by only setting 6 of them, the
  // choice of modes to use here is arbitrary.
  const auto evaluate_polynomial = [&coords](
      const std::array<double, 6>& coeffs) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    return DataVector{coeffs[0] + coeffs[1] * y + coeffs[2] * x * z +
                      coeffs[3] * x * y * z + coeffs[4] * square(y) * z +
                      coeffs[5] * square(x) * y * square(z)};
  };

  auto scalar =
      ScalarTag::type{{{evaluate_polynomial({{1., 0.5, 0.5, 0.2, 0.2, 0.1}})}}};

  // We skip one neighbor, lower_eta, to simulate an external boundary
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                     Variables<tmpl::list<ScalarTag>>,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_vars{};
  auto& vars_lower_xi =
      neighbor_vars[std::make_pair(Direction<3>::lower_xi(), ElementId<3>(1))];
  auto& vars_upper_xi =
      neighbor_vars[std::make_pair(Direction<3>::upper_xi(), ElementId<3>(2))];
  auto& vars_upper_eta =
      neighbor_vars[std::make_pair(Direction<3>::upper_eta(), ElementId<3>(4))];
  auto& vars_lower_zeta = neighbor_vars[std::make_pair(
      Direction<3>::lower_zeta(), ElementId<3>(5))];
  auto& vars_upper_zeta = neighbor_vars[std::make_pair(
      Direction<3>::upper_zeta(), ElementId<3>(6))];
  vars_lower_xi.initialize(mesh.number_of_grid_points());
  vars_upper_xi.initialize(mesh.number_of_grid_points());
  vars_upper_eta.initialize(mesh.number_of_grid_points());
  vars_lower_zeta.initialize(mesh.number_of_grid_points());
  vars_upper_zeta.initialize(mesh.number_of_grid_points());
  get<ScalarTag>(vars_lower_xi) =
      ScalarTag::type{{{evaluate_polynomial({{0.3, 0.2, 0.2, 0., 0., 0.1}})}}};
  get<ScalarTag>(vars_upper_xi) =
      ScalarTag::type{{{evaluate_polynomial({{2.5, 1., 0., 0., 1., 1.}})}}};
  get<ScalarTag>(vars_upper_eta) =
      ScalarTag::type{{{evaluate_polynomial({{1., 0.5, 0.5, 0.2, 0.2, 0.2}})}}};
  get<ScalarTag>(vars_lower_zeta) =
      ScalarTag::type{{{evaluate_polynomial({{1., 0.2, 0., 0., 0., 0.}})}}};
  get<ScalarTag>(vars_upper_zeta) =
      ScalarTag::type{{{evaluate_polynomial({{0.1, 0., 0.5, 0.2, 0.2, 0.2}})}}};

  SlopeLimiters::Weno_detail::reconstruct_from_weighted_sum<ScalarTag>(
      make_not_null(&scalar), mesh, neighbor_linear_weight, neighbor_vars);

  // Expected result computed in Mathematica by computing oscillation indicator
  // as in oscillation_indicator tests, then WENO weights, then superposition.
  auto expected_reconstructed_scalar = ScalarTag::type{{{evaluate_polynomial(
      {{1., 0.32663481881058243, 0.21186828015830592, 0.08447466846582166,
        0.08447504580872492, 0.04260655032204396}})}}};
  CHECK_ITERABLE_APPROX(scalar, expected_reconstructed_scalar);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.WenoHelpers.OscillationIndicator",
    "[SlopeLimiters][Unit]") {
  test_oscillation_indicator_1d();
  test_oscillation_indicator_2d();
  test_oscillation_indicator_3d();
}

// At the moment, this TEST_CASE cannot be merged with the OscillationIndicator
// TEST_CASE.
//
// This is because the function `oscillation_indicator` uses a static variable
// to cache the expensive computation of an intermediate result, which depends
// on the mesh. This caching is simplistic, and assumes a single mesh is used.
// This assumption is okay for our simple grids right now, but the caching will
// need to become more sophisticated when we want to handle more general
// domains.
//
// Until the caching is improved, the tests must choose between:
// - all tests (with a particular VolumeDim) use the same mesh, OR
// - tests use different meshes but are in different TEST_CASEs.
// We opt for the latter, as this provides a more rigorous test of the code.
SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.WenoHelpers.Reconstruction",
                  "[SlopeLimiters][Unit]") {
  test_reconstruction_1d();
  test_reconstruction_2d();
  test_reconstruction_3d();
}
