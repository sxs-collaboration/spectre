// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_exponential_filter(const double alpha, const unsigned half_power,
                             const double eps) noexcept {
  Approx local_approx = Approx::custom().epsilon(eps).scale(1.0);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  CAPTURE(eps);
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts < Spectral::maximum_number_of_points<BasisType>; ++num_pts) {
    CAPTURE(num_pts);
    const Mesh<1> mesh{num_pts, BasisType, QuadratureType};
    ModalVector initial_modal_coeffs(num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
      initial_modal_coeffs = i + 1.0;
    }
    const DataVector initial_nodal_coeffs =
        to_nodal_coefficients(initial_modal_coeffs, mesh);
    DataVector filtered_nodal_coeffs(num_pts);
    const Matrix filter_matrix =
        Spectral::filtering::exponential_filter(mesh, alpha, half_power);
    dgemv_('N', num_pts, num_pts, 1., filter_matrix.data(), num_pts,
           initial_nodal_coeffs.data(), 1, 0.0, filtered_nodal_coeffs.data(),
           1);
    const ModalVector filtered_modal_coeffs =
        to_modal_coefficients(filtered_nodal_coeffs, mesh);
    const double basis_order = num_pts - 1;
    for (size_t i = 0; i < num_pts; ++i) {
      CAPTURE(i);
      if (num_pts == 1) {
        // In the case of only 1 coefficient there should be no filtering.
        CHECK(filtered_modal_coeffs[i] == initial_modal_coeffs[i]);
      } else {
        CHECK(filtered_modal_coeffs[i] ==
              local_approx(initial_modal_coeffs[i] *
                           exp(-alpha * pow(i / basis_order, 2 * half_power))));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ExponentialFilter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  const std::vector<double> alphas{10.0, 20.0, 30.0, 40.0};
  const std::vector<unsigned> half_powers{2, 4, 8, 16};
  for (const double alpha : alphas) {
    for (const unsigned half_power : half_powers) {
      test_exponential_filter<Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto>(
          alpha, half_power, 2.0e-14);
      test_exponential_filter<Spectral::Basis::Legendre,
                              Spectral::Quadrature::Gauss>(alpha, half_power,
                                                           1.0e-12);
      test_exponential_filter<Spectral::Basis::Chebyshev,
                              Spectral::Quadrature::GaussLobatto>(
          alpha, half_power, 2.0e-14);
      test_exponential_filter<Spectral::Basis::Chebyshev,
                              Spectral::Quadrature::Gauss>(alpha, half_power,
                                                           1.0e-12);
    }
  }
}
}  // namespace
