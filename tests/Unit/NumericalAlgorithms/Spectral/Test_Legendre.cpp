// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/Legendre.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"

namespace {

double evaluate_by_summing_legendre_basis(const ModalVector& coefficients,
                                          const double x) {
  double sum = 0.0;
  for (size_t k = 0; k < coefficients.size(); ++k) {
    sum +=
        coefficients[k] *
        Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(k, x);
  }
  return sum;
}

template <size_t Dim>
void check_against_irregular(
    const std::array<size_t, Dim>& extents_array,
    const gsl::not_null<std::mt19937*> generator,
    std::uniform_real_distribution<>& coefficient_distribution,
    std::uniform_real_distribution<>& logical_distribution) {
  const size_t num_test_points = 100;
  const Mesh<Dim> mesh(extents_array, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
  const auto modal_coefficients = make_with_random_values<ModalVector>(
      generator, make_not_null(&coefficient_distribution),
      ModalVector{mesh.number_of_grid_points()});
  const auto logical_point =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::ElementLogical>>(
          generator, make_not_null(&logical_distribution), num_test_points);

  const DataVector nodal_coefficients =
      to_nodal_coefficients(modal_coefficients, mesh);
  const intrp::Irregular<Dim> interpolant(mesh, logical_point);
  const DataVector nodal_value = interpolant.interpolate(nodal_coefficients);
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (size_t i = 0; i < num_test_points; ++i) {
    tnsr::I<double, Dim, Frame::ElementLogical> single_logical_point{};
    for (size_t d = 0; d < Dim; ++d) {
      single_logical_point.get(d) = logical_point.get(d)[i];
    }
    CHECK(Spectral::evaluate_legendre_series<Dim>(modal_coefficients, mesh,
                                                  single_logical_point) ==
          custom_approx(nodal_value[i]));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Legendre.Series",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> coefficient_distribution(-2.0, 2.0);
  std::uniform_real_distribution<> logical_distribution(-1.0, 1.0);
  const auto evaluation_points = make_with_random_values<std::array<double, 5>>(
      make_not_null(&generator), make_not_null(&logical_distribution),
      std::array<double, 5>{});

  const auto cubic_coefficients = make_with_random_values<ModalVector>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      ModalVector{4});
  const Mesh<1> cubic_mesh({{cubic_coefficients.size()}},
                           Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto);
  for (const double x : evaluation_points) {
    CHECK(Spectral::evaluate_legendre_series<1>(
              cubic_coefficients, cubic_mesh,
              tnsr::I<double, 1, Frame::ElementLogical>{{{x}}}) ==
          approx(evaluate_by_summing_legendre_basis(cubic_coefficients, x)));
  }

  const auto higher_order = make_with_random_values<ModalVector>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      ModalVector{20});
  const Mesh<1> higher_mesh({{higher_order.size()}}, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  for (const double x : evaluation_points) {
    CHECK(Spectral::evaluate_legendre_series<1>(
              higher_order, higher_mesh,
              tnsr::I<double, 1, Frame::ElementLogical>{{{x}}}) ==
          approx(evaluate_by_summing_legendre_basis(higher_order, x)));
  }

  check_against_irregular<1>({{17}}, make_not_null(&generator),
                             coefficient_distribution, logical_distribution);
  check_against_irregular<2>({{20, 19}}, make_not_null(&generator),
                             coefficient_distribution, logical_distribution);
  check_against_irregular<3>({{12, 10, 7}}, make_not_null(&generator),
                             coefficient_distribution, logical_distribution);
}
