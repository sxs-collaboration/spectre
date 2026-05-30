// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/ModalVector.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Blas.hpp"

namespace {

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_exponential_filter(const double alpha, const unsigned half_power,
                             const double eps) {
  Approx local_approx = Approx::custom().epsilon(eps).scale(1.0);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  CAPTURE(eps);
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts <= Spectral::maximum_number_of_points<BasisType>; ++num_pts) {
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
    dgemv_('N', num_pts, num_pts, 1., filter_matrix.data(),
           filter_matrix.spacing(), initial_nodal_coeffs.data(), 1, 0.0,
           filtered_nodal_coeffs.data(), 1);
    const ModalVector filtered_modal_coeffs =
        to_modal_coefficients(filtered_nodal_coeffs, mesh);
    const double basis_order = static_cast<double>(num_pts) - 1;
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

void test_fourier_exponential_filter() {
  const Approx local_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);

  const std::vector<size_t> num_pts_list{1, 3, 5, 15};
  const std::vector<double> alphas{10.0, 20.0, 36.0};
  const std::vector<unsigned> half_powers{2, 4, 8};

  for (const size_t num_pts : num_pts_list) {
    CAPTURE(num_pts);
    const Mesh<1> mesh{num_pts, Spectral::Basis::Fourier,
                       Spectral::Quadrature::Equiangular};
    const DataVector& x = Spectral::collocation_points<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(num_pts);
    const size_t M = (num_pts - 1) / 2;

    for (const double alpha : alphas) {
      CAPTURE(alpha);
      for (const unsigned half_power : half_powers) {
        CAPTURE(half_power);
        const Matrix filter =
            Spectral::filtering::exponential_filter(mesh, alpha, half_power);

        for (size_t m = 1; m <= M; ++m) {
          CAPTURE(m);
          const double expected_weight =
              exp(-alpha * pow(static_cast<double>(m) / static_cast<double>(M),
                               2 * half_power));

          DataVector cos_vals = cos(static_cast<double>(m) * x);
          DataVector sin_vals = sin(static_cast<double>(m) * x);

          DataVector filtered_cos(num_pts, 0.0);
          DataVector filtered_sin(num_pts, 0.0);
          dgemv_('N', num_pts, num_pts, 1., filter.data(), filter.spacing(),
                 cos_vals.data(), 1, 0.0, filtered_cos.data(), 1);
          dgemv_('N', num_pts, num_pts, 1., filter.data(), filter.spacing(),
                 sin_vals.data(), 1, 0.0, filtered_sin.data(), 1);

          CHECK_ITERABLE_CUSTOM_APPROX(filtered_cos, expected_weight * cos_vals,
                                       local_approx);
          CHECK_ITERABLE_CUSTOM_APPROX(filtered_sin, expected_weight * sin_vals,
                                       local_approx);
        }
      }
    }
  }
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      Spectral::filtering::exponential_filter(
          Mesh<1>{4, Spectral::Basis::Fourier,
                  Spectral::Quadrature::Equiangular},
          2.0, 1),
      Catch::Matchers::ContainsSubstring("The Fourier basis is required to "
                                         "have an odd number of grid points"));
#endif  // SPECTRE_DEBUG
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ExponentialFilter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  const std::vector<double> alphas{10.0, 20.0, 30.0, 40.0};
  const std::vector<unsigned> half_powers{2, 4, 8, 16};
  for (const double alpha : alphas) {
    for (const unsigned half_power : half_powers) {
      test_exponential_filter<Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto>(
          alpha, half_power, 2.0e-12);
      test_exponential_filter<Spectral::Basis::Legendre,
                              Spectral::Quadrature::Gauss>(alpha, half_power,
                                                           1.0e-10);
      test_exponential_filter<Spectral::Basis::Chebyshev,
                              Spectral::Quadrature::GaussLobatto>(
          alpha, half_power, 2.0e-12);
      test_exponential_filter<Spectral::Basis::Chebyshev,
                              Spectral::Quadrature::Gauss>(alpha, half_power,
                                                           1.0e-10);
    }
  }
  test_fourier_exponential_filter();
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_zero_lowest_modes() {
  Approx local_approx = Approx::custom().epsilon(1.0e-11);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts <= Spectral::maximum_number_of_points<BasisType>; ++num_pts) {
    CAPTURE(num_pts);
    for (size_t number_of_modes_to_filter = 0;
         number_of_modes_to_filter < num_pts; ++number_of_modes_to_filter) {
      CAPTURE(number_of_modes_to_filter);
      const Mesh<1> mesh{num_pts, BasisType, QuadratureType};
      ModalVector initial_modal_coeffs(num_pts);
      for (size_t i = 0; i < num_pts; ++i) {
        initial_modal_coeffs = i + 1.0;
      }
      const DataVector initial_nodal_coeffs =
          to_nodal_coefficients(initial_modal_coeffs, mesh);
      DataVector filtered_nodal_coeffs(num_pts);
      const Matrix& filter_matrix = Spectral::filtering::zero_lowest_modes(
          mesh, number_of_modes_to_filter);
      dgemv_('N', num_pts, num_pts, 1., filter_matrix.data(),
             filter_matrix.spacing(), initial_nodal_coeffs.data(), 1, 0.0,
             filtered_nodal_coeffs.data(), 1);
      const ModalVector filtered_modal_coeffs =
          to_modal_coefficients(filtered_nodal_coeffs, mesh);
      for (size_t i = 0; i < num_pts; ++i) {
        CAPTURE(i);
        if (i < number_of_modes_to_filter) {
          CHECK(fabs(filtered_modal_coeffs[i]) < 1.0e-11);
        } else {
          CHECK(local_approx(filtered_modal_coeffs[i]) ==
                initial_modal_coeffs[i]);
        }
      }
    }
  }
}

void test_zero_lowest_modes_zernike_b1() {
  const Approx local_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (size_t num_pts = 2;
       num_pts <=
       Spectral::maximum_number_of_points<Spectral::Basis::ZernikeB1>;
       ++num_pts) {
    CAPTURE(num_pts);
    const Mesh<1> mesh{num_pts, Spectral::Basis::ZernikeB1,
                       Spectral::Quadrature::GaussRadauUpper};
    const DataVector& xi =
        Spectral::collocation_points<Spectral::Basis::ZernikeB1,
                                     Spectral::Quadrature::GaussRadauUpper>(
            num_pts);
    for (const auto parity : {Spectral::Parity::Even, Spectral::Parity::Odd}) {
      CAPTURE(parity);
      const size_t m = parity == Spectral::Parity::Even ? 0 : 1;
      const size_t n_modes = num_pts - m;
      for (size_t k = 0; k < n_modes; ++k) {
        CAPTURE(k);
        const Matrix& F =
            Spectral::filtering::zero_lowest_modes(mesh, k, parity);
        CHECK(F.rows() == num_pts);
        CHECK(F.columns() == num_pts);
        for (size_t mode = 0; mode < n_modes; ++mode) {
          DataVector nodal_pure_mode = Spectral::compute_basis_function_value<
              Spectral::Basis::ZernikeB1>(2 * mode + m, m, xi);
          DataVector filtered(num_pts, 0.0);
          dgemv_('N', num_pts, num_pts, 1.0, F.data(), F.spacing(),
                 nodal_pure_mode.data(), 1, 0.0, filtered.data(), 1);
          if (mode < k) {
            CHECK_ITERABLE_CUSTOM_APPROX(filtered, DataVector(num_pts, 0.0),
                                         local_approx);
          } else {
            CHECK_ITERABLE_CUSTOM_APPROX(filtered, nodal_pure_mode,
                                         local_approx);
          }
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ZeroLowestModesFilter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_zero_lowest_modes<Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto>();
  test_zero_lowest_modes<Spectral::Basis::Legendre,
                         Spectral::Quadrature::Gauss>();
  test_zero_lowest_modes<Spectral::Basis::Chebyshev,
                         Spectral::Quadrature::GaussLobatto>();
  test_zero_lowest_modes<Spectral::Basis::Chebyshev,
                         Spectral::Quadrature::Gauss>();
  test_zero_lowest_modes_zernike_b1();
}
}  // namespace
