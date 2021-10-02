// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {

void test_reconstruction_1d_impl(const Spectral::Quadrature quadrature) {
  CAPTURE(quadrature);
  const double neighbor_linear_weight = 0.005;
  const Mesh<1> mesh(5, Spectral::Basis::Legendre, quadrature);
  const auto coords = logical_coordinates(mesh);

  const auto evaluate_polynomial =
      [&coords](const std::array<double, 5>& coeffs) {
        const auto& x = get<0>(coords);
        return DataVector{coeffs[0] + coeffs[1] * x + coeffs[2] * square(x) +
                          coeffs[3] * cube(x) + coeffs[4] * pow<4>(x)};
      };

  DataVector local_data = evaluate_polynomial({{1., 2., 0., 0.5, 0.1}});
  // WENO reconstruction should preserve the mean, so expected = initial
  const double expected_local_mean = mean_value(local_data, mesh);

  const auto shift_data_to_local_mean =
      [&mesh, &expected_local_mean](const DataVector& neighbor_data) {
        return neighbor_data + expected_local_mean -
               mean_value(neighbor_data, mesh);
      };

  std::unordered_map<std::pair<Direction<1>, ElementId<1>>, DataVector,
                     boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data{};
  neighbor_data[std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1))] =
      shift_data_to_local_mean(evaluate_polynomial({{0., 1., 0., 1., 0.}}));
  neighbor_data[std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2))] =
      shift_data_to_local_mean(evaluate_polynomial({{0., 0., 1., 1., 2.}}));

  // Expected result computed in Mathematica by computing oscillation indicator
  // as in oscillation_indicator tests, then WENO weights, then superposition.
  const DataVector expected_reconstructed_data = evaluate_polynomial(
      {{1.0000250662809542, 1.9987344134217362, 3.25819395292328e-7,
        0.5006326303794342, 0.09987412556290375}});

  Limiters::Weno_detail::reconstruct_from_weighted_sum(
      make_not_null(&local_data), neighbor_linear_weight,
      Limiters::Weno_detail::DerivativeWeight::Unity, mesh, neighbor_data);
  CHECK(mean_value(local_data, mesh) == approx(expected_local_mean));
  CHECK_ITERABLE_APPROX(local_data, expected_reconstructed_data);
}

void test_reconstruction_1d() {
  INFO("Testing WENO reconstruction in 1D");
  test_reconstruction_1d_impl(Spectral::Quadrature::GaussLobatto);
  test_reconstruction_1d_impl(Spectral::Quadrature::Gauss);
}

void test_reconstruction_2d_impl(const Spectral::Quadrature quadrature) {
  CAPTURE(quadrature);
  const double neighbor_linear_weight = 0.001;
  const Mesh<2> mesh({{3, 3}}, Spectral::Basis::Legendre, quadrature);
  const auto coords = logical_coordinates(mesh);

  const auto evaluate_polynomial =
      [&coords](const std::array<double, 9>& coeffs) {
        const auto& x = get<0>(coords);
        const auto& y = get<1>(coords);
        return DataVector{
            coeffs[0] + coeffs[1] * x + coeffs[2] * square(x) +
            y * (coeffs[3] + coeffs[4] * x + coeffs[5] * square(x)) +
            square(y) * (coeffs[6] + coeffs[7] * x + coeffs[8] * square(x))};
      };

  DataVector local_data =
      evaluate_polynomial({{2., 1., 0., 1.5, 1., 0., 1., 0., 0.}});
  // WENO reconstruction should preserve the mean, so expected = initial
  const double expected_local_mean = mean_value(local_data, mesh);

  const auto shift_data_to_local_mean =
      [&mesh, &expected_local_mean](const DataVector& neighbor_data) {
        return neighbor_data + expected_local_mean -
               mean_value(neighbor_data, mesh);
      };

  std::unordered_map<std::pair<Direction<2>, ElementId<2>>, DataVector,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  neighbor_data[std::make_pair(Direction<2>::lower_xi(), ElementId<2>(1))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{0., 1., 0., 0., 1., 1., 0., 0., 0}}));
  neighbor_data[std::make_pair(Direction<2>::upper_xi(), ElementId<2>(2))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{0., 0., 1., 1., 2., 1., 0., 1., 1.}}));
  neighbor_data[std::make_pair(Direction<2>::lower_eta(), ElementId<2>(3))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{1., 0., 0., 0., 0.5, 0., 0., 0., 0.5}}));
  neighbor_data[std::make_pair(Direction<2>::upper_eta(), ElementId<2>(4))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{1., 0., 0., 0.5, 1., 0., 0., 0., 0.}}));

  // Expected result computed in Mathematica by computing oscillation indicator
  // as in oscillation_indicator tests, then WENO weights, then superposition.
  const DataVector expected_reconstructed_data = evaluate_polynomial(
      {{2.010056442214612, 0.9705606381771584, 0.000026246579961852654,
        1.4682459390241314, 0.9992393252122325, 0.0010535139634810797,
        0.9695333707936392, 0.000026246579961852654, 0.0008131679476911193}});

  Limiters::Weno_detail::reconstruct_from_weighted_sum(
      make_not_null(&local_data), neighbor_linear_weight,
      Limiters::Weno_detail::DerivativeWeight::Unity, mesh, neighbor_data);
  CHECK(mean_value(local_data, mesh) == approx(expected_local_mean));
  CHECK_ITERABLE_APPROX(local_data, expected_reconstructed_data);
}

void test_reconstruction_2d() {
  INFO("Testing WENO reconstruction in 2D");
  test_reconstruction_2d_impl(Spectral::Quadrature::GaussLobatto);
  test_reconstruction_2d_impl(Spectral::Quadrature::Gauss);
}

void test_reconstruction_3d_impl(const Spectral::Quadrature quadrature) {
  CAPTURE(quadrature);
  const double neighbor_linear_weight = 0.001;
  const Mesh<3> mesh({{3, 3, 3}}, Spectral::Basis::Legendre, quadrature);
  const auto coords = logical_coordinates(mesh);

  // 3D case has so many modes... so we simplify by only setting 6 of them, the
  // choice of modes to use here is arbitrary.
  const auto evaluate_polynomial =
      [&coords](const std::array<double, 6>& coeffs) {
        const auto& x = get<0>(coords);
        const auto& y = get<1>(coords);
        const auto& z = get<2>(coords);
        return DataVector{coeffs[0] + coeffs[1] * y + coeffs[2] * x * z +
                          coeffs[3] * x * y * z + coeffs[4] * square(y) * z +
                          coeffs[5] * square(x) * y * square(z)};
      };

  DataVector local_data = evaluate_polynomial({{1., 0.5, 0.5, 0.2, 0.2, 0.1}});
  // WENO reconstruction should preserve the mean, so expected = initial
  const double expected_local_mean = mean_value(local_data, mesh);

  const auto shift_data_to_local_mean =
      [&mesh, &expected_local_mean](const DataVector& neighbor_data) {
        return neighbor_data + expected_local_mean -
               mean_value(neighbor_data, mesh);
      };

  // We skip one neighbor, lower_eta, to simulate an external boundary
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>, DataVector,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data{};
  neighbor_data[std::make_pair(Direction<3>::lower_xi(), ElementId<3>(1))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{0.3, 0.2, 0.2, 0., 0., 0.1}}));
  neighbor_data[std::make_pair(Direction<3>::upper_xi(), ElementId<3>(2))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{2.5, 1., 0., 0., 1., 1.}}));
  neighbor_data[std::make_pair(Direction<3>::upper_eta(), ElementId<3>(4))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{1., 0.5, 0.5, 0.2, 0.2, 0.2}}));
  neighbor_data[std::make_pair(Direction<3>::lower_zeta(), ElementId<3>(5))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{1., 0.2, 0., 0., 0., 0.}}));
  neighbor_data[std::make_pair(Direction<3>::upper_zeta(), ElementId<3>(6))] =
      shift_data_to_local_mean(
          evaluate_polynomial({{0.1, 0., 0.5, 0.2, 0.2, 0.2}}));

  // Expected result computed in Mathematica by computing oscillation indicator
  // as in oscillation_indicator tests, then WENO weights, then superposition.
  const DataVector expected_reconstructed_data = evaluate_polynomial(
      {{1., 0.32663481881058243, 0.21186828015830592, 0.08447466846582166,
        0.08447504580872492, 0.04260655032204396}});

  Limiters::Weno_detail::reconstruct_from_weighted_sum(
      make_not_null(&local_data), neighbor_linear_weight,
      Limiters::Weno_detail::DerivativeWeight::Unity, mesh, neighbor_data);
  CHECK(mean_value(local_data, mesh) == approx(expected_local_mean));
  CHECK_ITERABLE_APPROX(local_data, expected_reconstructed_data);
}

void test_reconstruction_3d() {
  INFO("Testing WENO reconstruction in 3D");
  test_reconstruction_3d_impl(Spectral::Quadrature::GaussLobatto);
  test_reconstruction_3d_impl(Spectral::Quadrature::Gauss);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.Weno.Helpers",
                  "[Limiters][Unit]") {
  test_reconstruction_1d();
  test_reconstruction_2d();
  test_reconstruction_3d();
}
