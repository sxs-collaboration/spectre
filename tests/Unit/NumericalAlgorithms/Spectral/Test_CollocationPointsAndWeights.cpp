// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral {
namespace {
template <Basis basis, Quadrature quadrature>
void test() {
  CAPTURE(basis);
  CAPTURE(quadrature);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const auto& [xi, w] =
        compute_collocation_points_and_weights<basis, quadrature>(n);
    const size_t k_max = quadrature == Quadrature::Gauss ? n : n - 1;
    CAPTURE(k_max);
    for (size_t k = 1; k <= k_max; ++k) {
      CAPTURE(k);
      const auto f_k = compute_basis_function_value<basis>(k, xi);
      for (size_t j = 0; j < k; ++j) {
        CAPTURE(j);
        const auto f_j = compute_basis_function_value<basis>(j, xi);
        const double should_be_zero = sum(f_j * f_k * w);
        CHECK(should_be_zero == approx(0.0));
      }
    }
  }
}

template <Basis basis>
void test_radial_zernike() {
  // Needs its own test because radial zernike has two indices
  // Second index m must match parity of k, and satisfy m <= k
  // Checking \int_0^1 Q^m_n Q^m_{n'} \propto \delta_{n,n'} (no constraint on m)
  const auto quadrature = Quadrature::GaussRadauUpper;
  CAPTURE(basis);
  CAPTURE(quadrature);
  const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const auto& [xi, w] =
        compute_collocation_points_and_weights<basis, quadrature>(n);
    const size_t k_max = n - 1;
    CAPTURE(k_max);
    for (size_t k = 1; k <= k_max; ++k) {
      CAPTURE(k);
      for (size_t j = k % 2; j < k; j += 2) {
        CAPTURE(j);
        for (size_t m = k % 2; m <= j; m += 2) {
          CAPTURE(m);
          const auto f_m_k = compute_basis_function_value<basis>(k, m, xi);
          const auto f_m_j = compute_basis_function_value<basis>(j, m, xi);
          const double should_be_zero = sum(f_m_j * f_m_k * w);
          CHECK_ITERABLE_CUSTOM_APPROX(should_be_zero, 0.0, custom_approx);
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CollocationPointsAndWeights",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test<Basis::Legendre, Quadrature::Gauss>();
  test<Basis::Legendre, Quadrature::GaussLobatto>();
  test<Basis::Chebyshev, Quadrature::Gauss>();
  test<Basis::Chebyshev, Quadrature::GaussLobatto>();
  test<Basis::Fourier, Quadrature::Equiangular>();
  test_radial_zernike<Basis::ZernikeB1>();
  test_radial_zernike<Basis::ZernikeB2>();
  test_radial_zernike<Basis::ZernikeB3>();
}
}  // namespace Spectral
