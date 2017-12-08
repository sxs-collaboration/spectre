// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/Blas.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Linearize",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  // The start and end are chosen to give fast tests
  const size_t n_start = 2, n_end = 5;
  for (size_t nx = n_start; nx < n_end; ++nx) {
    const DataVector& x = Basis::lgl::collocation_points(nx);
    for (size_t ny = n_start; ny < n_end; ++ny) {
      const DataVector& y = Basis::lgl::collocation_points(ny);
      for (size_t nz = n_start; nz < n_end; ++nz) {
        const DataVector& z = Basis::lgl::collocation_points(nz);
        const Index<3> extents(nx, ny, nz);
        DataVector u(extents.product());
        for (IndexIterator<3> i(extents); i; ++i) {
          u[i.offset()] = exp(x[i()[0]]) * exp(y[i()[1]]) * exp(z[i()[2]]);
        }
        DataVector u_lin = linearize(u, extents);
        for (size_t d = 0; d < 3; ++d) {
          for (StripeIterator s(extents, d); s; ++s) {
            const Matrix& inv_v =
                Basis::lgl::grid_points_to_spectral_matrix(extents[d]);
            DataVector u_s(extents[d]);
            dgemv_('N', extents[d], extents[d], 1., inv_v.data(), extents[d],
                   u_lin.data() + s.offset(), s.stride(), 0.0,  // NOLINT
                   u_s.data(), 1);
            for (size_t i = 2; i < extents[d]; ++i) {
              CHECK(0.0 == approx(u_s[i]));
            }
          }
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.LinearizeALinearFunction",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  const size_t n_start = 2, n_end = 5;
  for (size_t nx = n_start; nx < n_end; ++nx) {
    const DataVector& x = Basis::lgl::collocation_points(nx);
    for (size_t ny = n_start; ny < n_end; ++ny) {
      const DataVector& y = Basis::lgl::collocation_points(ny);
      for (size_t nz = n_start; nz < n_end; ++nz) {
        const DataVector& z = Basis::lgl::collocation_points(nz);
        const Index<3> extents(nx, ny, nz);
        DataVector u(extents.product());
        for (IndexIterator<3> i(extents); i; ++i) {
          u[i.offset()] = 3*x[i()[0]]+5*y[i()[1]]+z[i()[2]];
        }
        const DataVector u_lin = linearize(u, extents);
        CHECK_ITERABLE_APPROX(u, u_lin);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.LinearizeInOneDim",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  // The start and end are chosen to give fast tests
  const size_t n_start = 3, n_end = 5;
  for (size_t nx = n_start; nx < n_end; ++nx) {
    const DataVector& x = Basis::lgl::collocation_points(nx);
    for (size_t ny = n_start; ny < n_end; ++ny) {
      const DataVector& y = Basis::lgl::collocation_points(ny);
      for (size_t nz = n_start; nz < n_end; ++nz) {
        const DataVector& z = Basis::lgl::collocation_points(nz);
        const Index<3> extents(nx, ny, nz);
        DataVector u(extents.product());
        for (IndexIterator<3> i(extents); i; ++i) {
          u[i.offset()] = exp(x[i()[0]]) * exp(y[i()[1]]) * exp(z[i()[2]]);
        }
        for (size_t d = 0; d < 3; ++d) {
          DataVector u_lin = linearize(u, extents, d);
          for (StripeIterator s(extents, d); s; ++s) {
            const Matrix& inv_v =
                Basis::lgl::grid_points_to_spectral_matrix(extents[d]);
            DataVector u_s(extents[d]);
            dgemv_('N', extents[d], extents[d], 1., inv_v.data(), extents[d],
                   u_lin.data() + s.offset(), s.stride(),  // NOLINT
                   0.0, u_s.data(), 1);
            for (size_t i = 2; i < extents[d]; ++i) {
              CHECK(0.0 == approx(u_s[i]));
            }
          }
        }
      }
    }
  }
}
