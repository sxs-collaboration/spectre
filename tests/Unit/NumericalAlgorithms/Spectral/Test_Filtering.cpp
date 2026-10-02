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
       num_pts <= Spectral::maximum_number_of_points<BasisType, QuadratureType>;
       ++num_pts) {
    CAPTURE(num_pts);
    const Mesh<1> mesh{num_pts, BasisType, QuadratureType};
    ModalVector initial_modal_coeffs(num_pts);
    for (size_t i = 0; i < num_pts; ++i) {
      initial_modal_coeffs[i] = static_cast<double>(i) + 1.0;
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

void test_half_fourier_exponential_filter() {
  const Approx local_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);

  const std::vector<size_t> num_pts_list{1, 2, 3, 4, 5, 12, 15};
  const std::vector<double> alphas{10.0, 20.0, 36.0};
  const std::vector<unsigned> half_powers{2, 4, 8};

  for (const size_t num_pts : num_pts_list) {
    CAPTURE(num_pts);
    const Mesh<1> mesh{num_pts, Spectral::Basis::HalfFourier,
                       Spectral::Quadrature::Equiangular};
    const DataVector& phi =
        Spectral::collocation_points<Spectral::Basis::HalfFourier,
                                     Spectral::Quadrature::Equiangular>(
            num_pts);

    for (const double alpha : alphas) {
      CAPTURE(alpha);
      for (const unsigned half_power : half_powers) {
        CAPTURE(half_power);
        const Matrix even_filter = Spectral::filtering::exponential_filter(
            mesh, alpha, half_power, Spectral::Parity::Even);
        const Matrix odd_filter = Spectral::filtering::exponential_filter(
            mesh, alpha, half_power, Spectral::Parity::Odd);
        CHECK(even_filter.rows() == num_pts);
        CHECK(odd_filter.rows() == num_pts);

        const auto filter_mode = [&num_pts](const Matrix& filter,
                                            const DataVector& nodal_values) {
          DataVector result(num_pts, 0.0);
          dgemv_('N', num_pts, num_pts, 1.0, filter.data(), filter.spacing(),
                 nodal_values.data(), 1, 0.0, result.data(), 1);
          return result;
        };

        // A single grid point cannot resolve anything beyond the constant, so
        // the filter is the identity
        if (num_pts == 1) {
          const DataVector constant_vals(num_pts, 1.0);
          CHECK_ITERABLE_CUSTOM_APPROX(filter_mode(even_filter, constant_vals),
                                       constant_vals, local_approx);
          CHECK_ITERABLE_CUSTOM_APPROX(filter_mode(odd_filter, constant_vals),
                                       constant_vals, local_approx);
          continue;
        }

        // The weight depends only on the wavenumber
        const double order = static_cast<double>(num_pts) - 1.0;
        const auto expected_weight = [&alpha, &half_power,
                                      &order](const size_t k) {
          return k == 0 ? 1.0
                        : exp(-alpha * pow(static_cast<double>(k) / order,
                                           2 * half_power));
        };

        // Even parity: the spectral space is cos(k phi) for k = 0, ..., N-1.
        // The constant mode is retained exactly.
        for (size_t k = 0; k < num_pts; ++k) {
          CAPTURE(k);
          const DataVector cos_vals = cos(static_cast<double>(k) * phi);
          CHECK_ITERABLE_CUSTOM_APPROX(filter_mode(even_filter, cos_vals),
                                       expected_weight(k) * cos_vals,
                                       local_approx);
        }

        // Odd parity: the spectral space is sin(k phi) for k = 1, ..., N. Modes
        // k = 1, ..., N-1 get exactly the same weight as the cosine of the same
        // wavenumber
        for (size_t k = 1; k < num_pts; ++k) {
          CAPTURE(k);
          const DataVector sin_vals = sin(static_cast<double>(k) * phi);
          CHECK_ITERABLE_CUSTOM_APPROX(filter_mode(odd_filter, sin_vals),
                                       expected_weight(k) * sin_vals,
                                       local_approx);
        }

        // The Nyquist sine is visible on the grid but must be removed.
        {
          const DataVector nyquist_sine =
              sin(static_cast<double>(num_pts) * phi);
          // Guard against a vacuous check: the Nyquist sine really is visible
          // on this grid.
          CHECK(max(abs(nyquist_sine)) > 0.5);
          CHECK_ITERABLE_CUSTOM_APPROX(filter_mode(odd_filter, nyquist_sine),
                                       DataVector(num_pts, 0.0), local_approx);
          // The Nyquist cosine, by contrast, is identically zero on the grid.
          CHECK_ITERABLE_CUSTOM_APPROX(cos(static_cast<double>(num_pts) * phi),
                                       DataVector(num_pts, 0.0), local_approx);
        }
      }
    }
  }

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(Spectral::filtering::exponential_filter(
                        Mesh<1>{5, Spectral::Basis::HalfFourier,
                                Spectral::Quadrature::Equiangular},
                        36.0, 8),
                    Catch::Matchers::ContainsSubstring(
                        "Need parity to be set to filter HalfFourier"));
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
  test_half_fourier_exponential_filter();
}

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_zero_lowest_modes() {
  Approx local_approx = Approx::custom().epsilon(1.0e-11);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts <= Spectral::maximum_number_of_points<BasisType, QuadratureType>;
       ++num_pts) {
    CAPTURE(num_pts);
    for (size_t number_of_modes_to_filter = 0;
         number_of_modes_to_filter < num_pts; ++number_of_modes_to_filter) {
      CAPTURE(number_of_modes_to_filter);
      const Mesh<1> mesh{num_pts, BasisType, QuadratureType};
      ModalVector initial_modal_coeffs(num_pts);
      for (size_t i = 0; i < num_pts; ++i) {
        initial_modal_coeffs[i] = static_cast<double>(i) + 1.0;
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
       Spectral::maximum_number_of_points<
           Spectral::Basis::ZernikeB1, Spectral::Quadrature::GaussRadauUpper>;
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

template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
void test_zero_highest_modes() {
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);
  for (size_t num_pts =
           Spectral::minimum_number_of_points<BasisType, QuadratureType>;
       num_pts <= Spectral::maximum_number_of_points<BasisType, QuadratureType>;
       ++num_pts) {
    CAPTURE(num_pts);
    // Zeroing modes is a round trip through the modal<->nodal transforms, so
    // the retained modes are reproduced and the dropped modes vanish to
    // roundoff. The roundoff accumulates with the number of points (chained
    // transforms) and scales with the data magnitude, whose largest modal
    // coefficient is `num_pts`.
    const Approx local_approx =
        Approx::custom().epsilon(1.0e-13).scale(static_cast<double>(num_pts));
    for (size_t number_of_modes_to_filter = 0;
         number_of_modes_to_filter < num_pts; ++number_of_modes_to_filter) {
      CAPTURE(number_of_modes_to_filter);
      const Mesh<1> mesh{num_pts, BasisType, QuadratureType};
      ModalVector initial_modal_coeffs(num_pts);
      for (size_t i = 0; i < num_pts; ++i) {
        initial_modal_coeffs[i] = static_cast<double>(i) + 1.0;
      }
      const DataVector initial_nodal_coeffs =
          to_nodal_coefficients(initial_modal_coeffs, mesh);
      DataVector filtered_nodal_coeffs(num_pts);
      const Matrix& filter_matrix = Spectral::filtering::zero_highest_modes(
          mesh, number_of_modes_to_filter);
      dgemv_('N', num_pts, num_pts, 1., filter_matrix.data(),
             filter_matrix.spacing(), initial_nodal_coeffs.data(), 1, 0.0,
             filtered_nodal_coeffs.data(), 1);
      const ModalVector filtered_modal_coeffs =
          to_modal_coefficients(filtered_nodal_coeffs, mesh);
      for (size_t i = 0; i < num_pts; ++i) {
        CAPTURE(i);
        if (i + number_of_modes_to_filter >= num_pts) {
          CHECK(filtered_modal_coeffs[i] == local_approx(0.0));
        } else {
          CHECK(local_approx(filtered_modal_coeffs[i]) ==
                initial_modal_coeffs[i]);
        }
      }
    }
  }
}

void test_fourier_zero_highest_modes() {
  Approx local_approx = approx;
  local_approx.scale(1.0);
  const std::vector<size_t> num_pts_list{1, 3, 5, 15};
  for (const size_t num_pts : num_pts_list) {
    CAPTURE(num_pts);
    const Mesh<1> mesh{num_pts, Spectral::Basis::Fourier,
                       Spectral::Quadrature::Equiangular};
    const DataVector& x = Spectral::collocation_points<
        Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(num_pts);
    const size_t M = (num_pts - 1) / 2;
    for (size_t number_of_modes_to_filter = 0; number_of_modes_to_filter <= M;
         ++number_of_modes_to_filter) {
      CAPTURE(number_of_modes_to_filter);
      const Matrix& filter = Spectral::filtering::zero_highest_modes(
          mesh, number_of_modes_to_filter);
      // The constant mode is always retained.
      const DataVector constant_vals(num_pts, 1.0);
      DataVector filtered_constant(num_pts, 0.0);
      dgemv_('N', num_pts, num_pts, 1., filter.data(), filter.spacing(),
             constant_vals.data(), 1, 0.0, filtered_constant.data(), 1);
      CHECK_ITERABLE_CUSTOM_APPROX(filtered_constant, constant_vals,
                                   local_approx);
      for (size_t m = 1; m <= M; ++m) {
        CAPTURE(m);
        const DataVector cos_vals = cos(static_cast<double>(m) * x);
        const DataVector sin_vals = sin(static_cast<double>(m) * x);
        DataVector filtered_cos(num_pts, 0.0);
        DataVector filtered_sin(num_pts, 0.0);
        dgemv_('N', num_pts, num_pts, 1., filter.data(), filter.spacing(),
               cos_vals.data(), 1, 0.0, filtered_cos.data(), 1);
        dgemv_('N', num_pts, num_pts, 1., filter.data(), filter.spacing(),
               sin_vals.data(), 1, 0.0, filtered_sin.data(), 1);
        // The top number_of_modes_to_filter m-modes are zeroed.
        const bool keep = m + number_of_modes_to_filter <= M;
        const DataVector expected_cos =
            keep ? cos_vals : DataVector(num_pts, 0.0);
        const DataVector expected_sin =
            keep ? sin_vals : DataVector(num_pts, 0.0);
        CHECK_ITERABLE_CUSTOM_APPROX(filtered_cos, expected_cos, local_approx);
        CHECK_ITERABLE_CUSTOM_APPROX(filtered_sin, expected_sin, local_approx);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ZeroHighestModesFilter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_zero_highest_modes<Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto>();
  test_zero_highest_modes<Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss>();
  test_zero_highest_modes<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::GaussLobatto>();
  test_zero_highest_modes<Spectral::Basis::Chebyshev,
                          Spectral::Quadrature::Gauss>();
  test_fourier_zero_highest_modes();
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      Spectral::filtering::zero_highest_modes(
          Mesh<1>{4, Spectral::Basis::Fourier,
                  Spectral::Quadrature::Equiangular},
          1),
      Catch::Matchers::ContainsSubstring("The Fourier basis is required to "
                                         "have an odd number of grid points"));
  CHECK_THROWS_WITH(Spectral::filtering::zero_highest_modes(
                        Mesh<1>{5, Spectral::Basis::Fourier,
                                Spectral::Quadrature::Equiangular},
                        3),
                    Catch::Matchers::ContainsSubstring("you cannot zero"));
  CHECK_THROWS_WITH(Spectral::filtering::zero_highest_modes(
                        Mesh<1>{4, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto},
                        4),
                    Catch::Matchers::ContainsSubstring("you cannot zero"));
#endif  // SPECTRE_DEBUG
}
}  // namespace
