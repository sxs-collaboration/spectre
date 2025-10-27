// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/FourierTestFunctions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Fourier.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"

namespace Spectral {
namespace {

void test_mode_number_to_storage_index() {
  for (size_t n = 0; n <= 3; ++n) {
    CAPTURE(n);
    CHECK(Fourier::modal_storage_index(Fourier::mode_at_storage_index(n)) == n);
  }
}

void test_modal_to_nodal_matrix() {
  constexpr Basis basis = Basis::Fourier;
  constexpr Quadrature quadrature = Quadrature::Equiangular;
  for (size_t n = 1; n <= 81; n += 1) {
    CAPTURE(n);
    const DataVector& x = collocation_points<basis, quadrature>(n);
    const Matrix& m_to_n = modal_to_nodal_matrix<basis, quadrature>(n);
    const Matrix& n_to_m = nodal_to_modal_matrix<basis, quadrature>(n);
    for (size_t j = 0; j < n; ++j) {
      CAPTURE(j);
      DataVector u_k{n, 0.0};
      u_k[j] = 1.0;
      const DataVector u = apply_matrix(m_to_n, u_k);
      const int k = Fourier::mode_at_storage_index(j);
      const DataVector expected_u =
          k < 0 ? DataVector{sin(-k * x)} : DataVector{cos(k * x)};
      CHECK_ITERABLE_APPROX(u, expected_u);
      const DataVector computed_u_k = apply_matrix(n_to_m, u);
      CHECK_ITERABLE_APPROX(u_k, computed_u_k);
    }
  }
}

void test_definite_integral(
    const DataVector& u, const DataVector& weights,
    const FourierTestFunctions::ProductOfPolynomials& f) {
  const double result = ddot_(u.size(), weights.data(), 1, u.data(), 1);
  CHECK(f.definite_integral() == approx(result));
}

void test_derivative(const DataVector& u, const Matrix& m,
                     const FourierTestFunctions::ProductOfPolynomials& f,
                     const DataVector& phi) {
  const auto custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  const DataVector expected_du = f.df_dph(phi);
  const DataVector du = apply_matrix(m, u);
  CHECK_ITERABLE_CUSTOM_APPROX(du, expected_du, custom_approx);
}

template <typename T>
void test_interpolation(const DataVector& u, const T& target_points,
                        const Matrix& m,
                        const FourierTestFunctions::ProductOfPolynomials& f) {
  const auto custom_approx = Approx::custom().epsilon(5.0e-11).scale(1.0);
  const DataVector u_target = apply_matrix(m, u);
  for (size_t i = 0; i < get_size(target_points); ++i) {
    CHECK(get_element(u_target, i) ==
          custom_approx(f(get_element(target_points, i))));
  }
}

void test_transforms(const DataVector& u, const Matrix& modal_to_nodal_matrix,
                     const Matrix& nodal_to_modal_matrix,
                     const FourierTestFunctions::ProductOfPolynomials& f) {
  const DataVector expected_u_k = f.modes();
  const DataVector u_k = apply_matrix(nodal_to_modal_matrix, u);
  for (size_t i = 0; i < expected_u_k.size(); ++i) {
    CHECK(u_k[i] == approx(expected_u_k[i]));
  }
  const DataVector transformed_u = apply_matrix(modal_to_nodal_matrix, u_k);
  CHECK_ITERABLE_APPROX(u, transformed_u);
}

void test() {
  constexpr Basis basis = Basis::Fourier;
  constexpr Quadrature quadrature = Quadrature::Equiangular;
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  const double phi_target = phi_distribution(generator);
  const auto target_points = make_with_random_values<DataVector>(
      make_not_null(&generator), make_not_null(&phi_distribution), 5_st);
  CAPTURE(phi_target);
  for (size_t num_points = 1; num_points < 65; ++num_points) {
    CAPTURE(num_points);
    const DataVector phi = collocation_points<basis, quadrature>(num_points);
    const Matrix dm = differentiation_matrix<basis, quadrature>(num_points);
    const DataVector integration_weights =
        quadrature_weights<basis, quadrature>(num_points);
    const Matrix interp_matrix =
        interpolation_matrix<basis, quadrature>(num_points, phi_target);
    const Matrix interp_matrix_dv =
        interpolation_matrix<basis, quadrature>(num_points, target_points);
    const Matrix& m_to_n = modal_to_nodal_matrix<basis, quadrature>(num_points);
    const Matrix& n_to_m = nodal_to_modal_matrix<basis, quadrature>(num_points);
    const size_t k_max = num_points / 2;
    CAPTURE(k_max);
    for (size_t pow_nx = 0; pow_nx < k_max; ++pow_nx) {
      CAPTURE(pow_nx);
      for (size_t pow_ny = 0; pow_ny < k_max - pow_nx; ++pow_ny) {
        CAPTURE(pow_ny);
        const FourierTestFunctions::ProductOfPolynomials f{pow_nx, pow_ny};
        const DataVector u = f(phi);
        CAPTURE(u);
        test_definite_integral(u, integration_weights, f);
        test_interpolation(u, phi_target, interp_matrix, f);
        test_interpolation(u, target_points, interp_matrix_dv, f);
        test_derivative(u, dm, f, phi);
        test_transforms(u, m_to_n, n_to_m, f);
      }
    }
  }
}

DataVector expected_modes(const size_t pow_nx, const size_t pow_ny) {
  if (pow_nx == 0 and pow_ny == 0) {
    return {1.0};
  } else if (pow_nx == 1 and pow_ny == 0) {
    return {0.0, 1.0, 0.0};
  } else if (pow_nx == 0 and pow_ny == 1) {
    return {0.0, 0.0, 1.0};
  } else if (pow_nx == 2 and pow_ny == 0) {
    return {0.5, 0.0, 0.0, 0.5, 0.0};
  } else if (pow_nx == 1 and pow_ny == 1) {
    return {0.0, 0.0, 0.0, 0.0, 0.5};
  } else if (pow_nx == 0 and pow_ny == 2) {
    return {0.5, 0.0, 0.0, -0.5, 0.0};
  } else if (pow_nx == 3 and pow_ny == 0) {
    return {0.0, 0.75, 0.0, 0.0, 0.0, 0.25, 0.0};
  } else if (pow_nx == 2 and pow_ny == 1) {
    return {0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25};
  } else if (pow_nx == 1 and pow_ny == 2) {
    return {0.0, 0.25, 0.0, 0.0, 0.0, -0.25, 0.0};
  } else if (pow_nx == 0 and pow_ny == 3) {
    return {0.0, 0.0, 0.75, 0.0, 0.0, 0.0, -0.25};
  } else if (pow_nx == 4 and pow_ny == 0) {
    return {0.375, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.125, 0.0};
  } else if (pow_nx == 3 and pow_ny == 1) {
    return {0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.125};
  } else if (pow_nx == 2 and pow_ny == 2) {
    return {0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.125, 0.0};
  } else if (pow_nx == 1 and pow_ny == 3) {
    return {0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, -0.125};
  } else if (pow_nx == 0 and pow_ny == 4) {
    return {0.375, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.125, 0.0};
  }
  return DataVector{};
}

void test_modes() {
  for (size_t k_max = 0; k_max < 5; ++k_max) {
    for (size_t pow_nx = 0; pow_nx <= k_max; ++pow_nx) {
      CAPTURE(pow_nx);
      for (size_t pow_ny = 0; pow_ny <= k_max - pow_nx; ++pow_ny) {
        CAPTURE(pow_ny);
        const FourierTestFunctions::ProductOfPolynomials f{pow_nx, pow_ny};
        CHECK_ITERABLE_APPROX(expected_modes(pow_nx, pow_ny), f.modes());
      }
    }
  }
}
}  // namespace

// [[Timeout, 10]]
SPECTRE_TEST_CASE("Unit.Numerical.Spectral.BasisFunctions.Fourier",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_mode_number_to_storage_index();
  test_modal_to_nodal_matrix();
  test();
  test_modes();
}
}  // namespace Spectral
