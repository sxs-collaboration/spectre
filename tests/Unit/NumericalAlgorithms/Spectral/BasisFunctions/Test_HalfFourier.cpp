// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <numbers>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace {

void test_collocation_points_and_weights() {
  // phi_j = (j + 1/2) * pi / N, w_j = pi / N for all j
  const Approx custom_approx = Approx::custom().epsilon(1.0e-14).scale(1.0);
  for (size_t n = 1; n <= 10; ++n) {
    CAPTURE(n);
    const DataVector& phi =
        collocation_points<Basis::HalfFourier, Quadrature::Equiangular>(n);
    const DataVector& weights =
        quadrature_weights<Basis::HalfFourier, Quadrature::Equiangular>(n);
    REQUIRE(phi.size() == n);
    REQUIRE(weights.size() == n);
    const double pi_over_n = std::numbers::pi / static_cast<double>(n);
    for (size_t j = 0; j < n; ++j) {
      CHECK(phi[j] ==
            custom_approx(pi_over_n * (static_cast<double>(j) + 0.5)));
      CHECK(weights[j] == custom_approx(pi_over_n));
    }
  }
}

void test_differentiation_matrices() {
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (size_t n = 1; n <= 8; ++n) {
    CAPTURE(n);
    const DataVector& phi =
        collocation_points<Basis::HalfFourier, Quadrature::Equiangular>(n);
    const Matrix& D_even =
        differentiation_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
            n, Parity::Even);
    const Matrix& D_odd =
        differentiation_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
            n, Parity::Odd);

    REQUIRE(D_even.rows() == n);
    REQUIRE(D_even.columns() == n);
    REQUIRE(D_odd.rows() == n);
    REQUIRE(D_odd.columns() == n);

    {
      INFO("D_even applied to constant (k=0): should give zero");
      const DataVector one(n, 1.0);
      const DataVector result = apply_matrix(D_even, one);
      const DataVector expected(n, 0.0);
      CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
    }

    {
      INFO("D_even * cos(k*phi) = -k * sin(k*phi) for k = 1, ..., n-1");
      for (size_t k = 1; k < n; ++k) {
        CAPTURE(k);
        const DataVector cos_k = cos(static_cast<double>(k) * phi);
        const DataVector expected =
            -static_cast<double>(k) * sin(static_cast<double>(k) * phi);
        const DataVector result = apply_matrix(D_even, cos_k);
        CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
      }
    }

    {
      INFO("D_odd * sin(k*phi) = k * cos(k*phi) for k = 1, ..., n");
      for (size_t k = 1; k <= n; ++k) {
        CAPTURE(k);
        const DataVector sin_k = sin(static_cast<double>(k) * phi);
        const DataVector expected =
            static_cast<double>(k) * cos(static_cast<double>(k) * phi);
        const DataVector result = apply_matrix(D_odd, sin_k);
        CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
      }
    }

    {
      INFO("Antisymmetry relation: D_even[i,j] = -D_odd[j,i]");
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          CAPTURE(i);
          CAPTURE(j);
          CHECK(D_even(i, j) == custom_approx(-D_odd(j, i)));
        }
      }
    }
  }
}

void test_interpolation_matrices() {
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> phi_dist(0.0, std::numbers::pi);
  const auto target_pts = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&phi_dist), 5_st);

  for (size_t n = 1; n <= 8; ++n) {
    CAPTURE(n);
    const DataVector& phi =
        collocation_points<Basis::HalfFourier, Quadrature::Equiangular>(n);

    {
      INFO("Identity at source points");
      const Matrix I_even =
          interpolation_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
              n, phi, Parity::Even);
      const Matrix I_odd =
          interpolation_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
              n, phi, Parity::Odd);
      REQUIRE(I_even.rows() == n);
      REQUIRE(I_even.columns() == n);
      REQUIRE(I_odd.rows() == n);
      REQUIRE(I_odd.columns() == n);
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          CHECK(I_even(i, j) == custom_approx(i == j ? 1.0 : 0.0));
          CHECK(I_odd(i, j) == custom_approx(i == j ? 1.0 : 0.0));
        }
      }
    }

    {
      INFO("Even interpolation");
      const Matrix I_even =
          interpolation_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
              n, target_pts, Parity::Even);
      REQUIRE(I_even.rows() == 5);
      REQUIRE(I_even.columns() == n);
      for (size_t k = 0; k < n; ++k) {
        CAPTURE(k);
        const DataVector u = cos(static_cast<double>(k) * phi);
        const DataVector expected = cos(static_cast<double>(k) * target_pts);
        const DataVector result = apply_matrix(I_even, u);
        CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
      }
    }

    {
      INFO("Odd interpolation");
      const Matrix I_odd =
          interpolation_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
              n, target_pts, Parity::Odd);
      REQUIRE(I_odd.rows() == 5);
      REQUIRE(I_odd.columns() == n);
      for (size_t k = 1; k <= n; ++k) {
        CAPTURE(k);
        const DataVector u = sin(static_cast<double>(k) * phi);
        const DataVector expected = sin(static_cast<double>(k) * target_pts);
        const DataVector result = apply_matrix(I_odd, u);
        CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
      }
    }
  }
}

void test_modal_to_nodal_matrices() {
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (size_t n = 1; n <= 8; ++n) {
    CAPTURE(n);
    const DataVector& phi =
        collocation_points<Basis::HalfFourier, Quadrature::Equiangular>(n);

    {
      INFO("Even parity");
      const Matrix& mtn =
          modal_to_nodal_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
              n, Parity::Even);
      const Matrix& ntm =
          nodal_to_modal_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
              n, Parity::Even);
      REQUIRE(mtn.rows() == n);
      REQUIRE(mtn.columns() == n);
      REQUIRE(ntm.rows() == n);
      REQUIRE(ntm.columns() == n);

      // Applying NTM to cos(k*phi) should give the k-th unit vector
      for (size_t k = 0; k < n; ++k) {
        CAPTURE(k);
        const DataVector u = cos(static_cast<double>(k) * phi);
        const DataVector coeffs = apply_matrix(ntm, u);
        for (size_t i = 0; i < n; ++i) {
          CHECK(coeffs[i] == custom_approx(i == k ? 1.0 : 0.0));
        }
      }

      // Applying MTN to a unit vector should reconstruct cos(k*phi)
      for (size_t k = 0; k < n; ++k) {
        CAPTURE(k);
        DataVector e_k(n, 0.0);
        e_k[k] = 1.0;
        const DataVector nodal = apply_matrix(mtn, e_k);
        const DataVector expected = cos(static_cast<double>(k) * phi);
        CHECK_ITERABLE_CUSTOM_APPROX(nodal, expected, custom_approx);
      }

      // Round-trip: NTM * MTN = I
      const Matrix round_trip_ntm_mtm = ntm * mtn;
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          CHECK(round_trip_ntm_mtm(i, j) == custom_approx(i == j ? 1.0 : 0.0));
        }
      }
    }

    {
      INFO("Odd parity");
      const Matrix& mtn =
          modal_to_nodal_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
              n, Parity::Odd);
      const Matrix& ntm =
          nodal_to_modal_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
              n, Parity::Odd);
      REQUIRE(mtn.rows() == n);
      REQUIRE(mtn.columns() == n);
      REQUIRE(ntm.rows() == n);
      REQUIRE(ntm.columns() == n);

      // Applying NTM to sin(k*phi) for k=1,...,N should give the (k-1)-th unit
      // vector
      for (size_t k = 1; k <= n; ++k) {
        CAPTURE(k);
        const DataVector u = sin(static_cast<double>(k) * phi);
        const DataVector coeffs = apply_matrix(ntm, u);
        for (size_t i = 0; i < n; ++i) {
          CHECK(coeffs[i] == custom_approx(i == k - 1 ? 1.0 : 0.0));
        }
      }

      // Applying MTN to a unit vector should reconstruct sin(k*phi)
      for (size_t k = 1; k <= n; ++k) {
        CAPTURE(k);
        DataVector e_k(n, 0.0);
        e_k[k - 1] = 1.0;
        const DataVector nodal = apply_matrix(mtn, e_k);
        const DataVector expected = sin(static_cast<double>(k) * phi);
        CHECK_ITERABLE_CUSTOM_APPROX(nodal, expected, custom_approx);
      }

      // Round-trip: NTM * MTN = I
      const Matrix round_trip_ntm_mtm = ntm * mtn;
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          CHECK(round_trip_ntm_mtm(i, j) == custom_approx(i == j ? 1.0 : 0.0));
        }
      }
    }
  }
#ifdef SPECTRE_DEBUG
  const DataVector& phi =
      collocation_points<Basis::HalfFourier, Quadrature::Equiangular>(4);
  CHECK_THROWS_WITH(
      (modal_to_nodal_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
          4, Parity::Uninitialized)),
      Catch::Matchers::ContainsSubstring(
          "Tried to use a parity-based function without a definite parity"));
  CHECK_THROWS_WITH(
      (nodal_to_modal_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
          4, Parity::Uninitialized)),
      Catch::Matchers::ContainsSubstring(
          "Tried to use a parity-based function without a definite parity"));
  CHECK_THROWS_WITH(
      (interpolation_matrix<Basis::HalfFourier, Quadrature::Equiangular>(
          4, phi, Parity::Uninitialized)),
      Catch::Matchers::ContainsSubstring(
          "Parity must be set to either Even or Odd"));
#endif
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.BasisFunctions.HalfFourier",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_collocation_points_and_weights();
  test_differentiation_matrices();
  test_interpolation_matrices();
  test_modal_to_nodal_matrices();
}
}  // namespace Spectral
