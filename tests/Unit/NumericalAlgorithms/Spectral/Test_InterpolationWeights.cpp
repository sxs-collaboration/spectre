// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationWeights.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_fornberg_matrix(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  for (size_t n_target_points = 1; n_target_points < 101;
       n_target_points += 11) {
    const auto x_target = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), n_target_points);
    for (size_t n_source_points = 4; n_source_points < 20; ++n_source_points) {
      // Check that we get the same matrices as Spectral::interpolation_matrix
      // for collocation points of existing Basis and Quadrature
      for (const auto basis :
           std::array{Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}) {
        for (const auto quadrature :
             std::array{Spectral::Quadrature::Gauss,
                        Spectral::Quadrature::GaussLobatto}) {
          const Mesh<1> mesh{n_source_points, basis, quadrature};
          const auto xi = logical_coordinates(mesh);
          const Matrix spectral_matrix =
              Spectral::interpolation_matrix(mesh, x_target);
          const Matrix fornberg_matrix =
              Spectral::fornberg_interpolation_matrix(x_target, get<0>(xi));
          CHECK(fornberg_matrix.rows() == n_target_points);
          CHECK(fornberg_matrix.columns() == n_source_points);
          CHECK(spectral_matrix == fornberg_matrix);
          if (n_target_points == 1) {
            CHECK(Spectral::fornberg_interpolation_matrix(
                      x_target[0], get<0>(xi)) == fornberg_matrix);
          }
        }
      }
    }
  }
}

void test_fornberg_derivative_coefficients(
    const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  const size_t max_derivative = 4;
  std::array<DataVector, max_derivative + 1> fornberg_weights{};
  const size_t n_target_points = 5;
  const auto x_targets = make_with_random_values<DataVector>(
      generator, make_not_null(&xi_distribution), n_target_points);
  for (size_t n_source_points = 3; n_source_points < 5; ++n_source_points) {
    const Mesh<1> mesh{n_source_points, Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss};
    const auto xi = logical_coordinates(mesh);
    for (auto x_target : x_targets) {
      Spectral::fornberg_derivative_interpolation_weights<max_derivative>(
          make_not_null(&fornberg_weights), x_target, get<0>(xi));
      double coefficient = 1.0;
      auto power = static_cast<int>(n_source_points - 1);
      for (size_t derivative = 0; derivative <= n_source_points; ++derivative) {
        const DataVector function = pow(get<0>(xi), power);
        const double result = std::inner_product(
            function.begin(), function.end(),
            gsl::at(fornberg_weights, derivative).begin(), 0.0);
        CHECK(approx(result) ==
              coefficient *
                  pow(x_target, power - static_cast<int>(derivative)));
        coefficient *= power - static_cast<int>(derivative);
      }
    }
  }
}

DataVector f_periodic(const DataVector& x) { return sin(M_PI * cos(x)); }

void test_fourier_matrix(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> phi_distribution(0.0, 2 * M_PI);
  const size_t n_source_points = 48;
  const DataVector x_source = get<0>(logical_coordinates(
      Mesh<1>{n_source_points, Spectral::Basis::SphericalHarmonic,
              Spectral::Quadrature::Equiangular}));
  DataVector f_source = f_periodic(x_source);
  for (size_t n_target_points = 1; n_target_points < 101;
       n_target_points += 11) {
    const auto x_target = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), n_target_points);
    const DataVector f_target = f_periodic(x_target);
    const Matrix m =
        Spectral::fourier_interpolation_matrix(x_target, n_source_points);
    DataVector f_interp{n_target_points};
    dgemv_('N', n_target_points, n_source_points, 1.0, m.data(),
           n_target_points, f_source.data(), 1, 0.0, f_interp.data(), 1);
    for (size_t k = 0; k < n_target_points; ++k) {
      CHECK_THAT(f_interp[k], Catch::Matchers::WithinAbs(f_target[k], 1.e-13));
    }
    if (n_target_points == 1) {
      CHECK(Spectral::fourier_interpolation_matrix(x_target[0],
                                                   n_source_points) == m);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.Weights",
                  "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test_fornberg_matrix(make_not_null(&generator));
  test_fornberg_derivative_coefficients(make_not_null(&generator));
  test_fourier_matrix(make_not_null(&generator));
}
