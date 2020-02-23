// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Variables

namespace {

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
  // WENO reconstruction should preserve the mean, so expected = initial
  const double expected_scalar_mean = mean_value(get(scalar), mesh);

  const auto shift_data_to_local_mean =
      [&mesh, &expected_scalar_mean ](const ScalarTag::type& data) noexcept {
    auto result = data;
    get(result) += expected_scalar_mean - mean_value(get(data), mesh);
    return result;
  };

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
  get<ScalarTag>(vars_lower_xi) = shift_data_to_local_mean(
      ScalarTag::type{{{evaluate_polynomial({{0., 1., 0., 1., 0.}})}}});
  get<ScalarTag>(vars_upper_xi) = shift_data_to_local_mean(
      ScalarTag::type{{{evaluate_polynomial({{0., 0., 1., 1., 2.}})}}});

  Limiters::Weno_detail::reconstruct_from_weighted_sum<ScalarTag>(
      make_not_null(&scalar), mesh, neighbor_linear_weight, neighbor_vars,
      Limiters::Weno_detail::DerivativeWeight::Unity);

  CHECK(mean_value(get(scalar), mesh) == approx(expected_scalar_mean));

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
  // WENO reconstruction should preserve the mean, so expected = initial
  const std::array<double, 2> expected_vector_means = {
      {mean_value(get<0>(vector), mesh), mean_value(get<1>(vector), mesh)}};

  const auto shift_data_to_local_mean = [&mesh, &expected_vector_means ](
      const VectorTag<2>::type& data) noexcept {
    auto result = data;
    get<0>(result) += expected_vector_means[0] - mean_value(get<0>(data), mesh);
    get<1>(result) += expected_vector_means[1] - mean_value(get<1>(data), mesh);
    return result;
  };

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
  get<VectorTag<2>>(vars_lower_xi) =
      shift_data_to_local_mean(VectorTag<2>::type{
          {{evaluate_polynomial({{0., 1., 0., 0., 1., 1., 0., 0., 0}}),
            evaluate_polynomial({{1., 0.1, 0., 0.1, 0., 0., 0., 0.}})}}});
  get<VectorTag<2>>(vars_upper_xi) =
      shift_data_to_local_mean(VectorTag<2>::type{
          {{evaluate_polynomial({{0., 0., 1., 1., 2., 1., 0., 1., 1.}}),
            evaluate_polynomial({{0., 1., 0., 1., 0., 0., 0., 0., 0.}})}}});
  get<VectorTag<2>>(vars_lower_eta) =
      shift_data_to_local_mean(VectorTag<2>::type{
          {{evaluate_polynomial({{1., 0., 0., 0., 0.5, 0., 0., 0., 0.5}}),
            evaluate_polynomial({{1., 0., 1., 1., 0., 0., 0., 0.}})}}});
  get<VectorTag<2>>(vars_upper_eta) =
      shift_data_to_local_mean(VectorTag<2>::type{
          {{evaluate_polynomial({{1., 0., 0., 0.5, 1., 0., 0., 0., 0.}}),
            evaluate_polynomial({{0., 0., 0., 1., 0., 0., 1., 1., 1.}})}}});

  Limiters::Weno_detail::reconstruct_from_weighted_sum<VectorTag<2>>(
      make_not_null(&vector), mesh, neighbor_linear_weight, neighbor_vars,
      Limiters::Weno_detail::DerivativeWeight::Unity);

  CHECK(mean_value(get<0>(vector), mesh) == approx(expected_vector_means[0]));
  CHECK(mean_value(get<1>(vector), mesh) == approx(expected_vector_means[1]));

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
  // WENO reconstruction should preserve the mean, so expected = initial
  const double expected_scalar_mean = mean_value(get(scalar), mesh);

  const auto shift_data_to_local_mean =
      [&mesh, &expected_scalar_mean ](const ScalarTag::type& data) noexcept {
    auto result = data;
    get(result) += expected_scalar_mean - mean_value(get(data), mesh);
    return result;
  };

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
  get<ScalarTag>(vars_lower_xi) = shift_data_to_local_mean(
      ScalarTag::type{{{evaluate_polynomial({{0.3, 0.2, 0.2, 0., 0., 0.1}})}}});
  get<ScalarTag>(vars_upper_xi) = shift_data_to_local_mean(
      ScalarTag::type{{{evaluate_polynomial({{2.5, 1., 0., 0., 1., 1.}})}}});
  get<ScalarTag>(vars_upper_eta) = shift_data_to_local_mean(ScalarTag::type{
      {{evaluate_polynomial({{1., 0.5, 0.5, 0.2, 0.2, 0.2}})}}});
  get<ScalarTag>(vars_lower_zeta) = shift_data_to_local_mean(
      ScalarTag::type{{{evaluate_polynomial({{1., 0.2, 0., 0., 0., 0.}})}}});
  get<ScalarTag>(vars_upper_zeta) = shift_data_to_local_mean(ScalarTag::type{
      {{evaluate_polynomial({{0.1, 0., 0.5, 0.2, 0.2, 0.2}})}}});

  Limiters::Weno_detail::reconstruct_from_weighted_sum<ScalarTag>(
      make_not_null(&scalar), mesh, neighbor_linear_weight, neighbor_vars,
      Limiters::Weno_detail::DerivativeWeight::Unity);

  CHECK(mean_value(get(scalar), mesh) == approx(expected_scalar_mean));

  // Expected result computed in Mathematica by computing oscillation indicator
  // as in oscillation_indicator tests, then WENO weights, then superposition.
  auto expected_reconstructed_scalar = ScalarTag::type{{{evaluate_polynomial(
      {{1., 0.32663481881058243, 0.21186828015830592, 0.08447466846582166,
        0.08447504580872492, 0.04260655032204396}})}}};
  CHECK_ITERABLE_APPROX(scalar, expected_reconstructed_scalar);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.Weno.Helpers",
                  "[Limiters][Unit]") {
  test_reconstruction_1d();
  test_reconstruction_2d();
  test_reconstruction_3d();
}
