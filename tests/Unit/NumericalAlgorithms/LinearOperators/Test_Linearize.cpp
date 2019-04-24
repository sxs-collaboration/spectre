// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Linearize",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  // The start and end are chosen to give fast tests
  const size_t n_start = 2, n_end = 5;
  for (size_t nx = n_start; nx < n_end; ++nx) {
    for (size_t ny = n_start; ny < n_end; ++ny) {
      for (size_t nz = n_start; nz < n_end; ++nz) {
        const Mesh<3> mesh{{{nx, ny, nz}},
                           Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
        const DataVector& x =
            Spectral::collocation_points(mesh.slice_through(0));
        const DataVector& y =
            Spectral::collocation_points(mesh.slice_through(1));
        const DataVector& z =
            Spectral::collocation_points(mesh.slice_through(2));
        DataVector u(mesh.number_of_grid_points());
        for (IndexIterator<3> i(mesh.extents()); i; ++i) {
          u[i.collapsed_index()] =
              exp(x[i()[0]]) * exp(y[i()[1]]) * exp(z[i()[2]]);
        }
        DataVector u_lin = linearize(u, mesh);
        for (size_t d = 0; d < 3; ++d) {
          for (StripeIterator s(mesh.extents(), d); s; ++s) {
            const Matrix& inv_v =
                Spectral::nodal_to_modal_matrix(mesh.slice_through(d));
            const auto slice_points = mesh.extents(d);
            DataVector u_s(slice_points);
            dgemv_('N', slice_points, slice_points, 1., inv_v.data(),
                   slice_points, u_lin.data() + s.offset(),  // NOLINT
                   s.stride(), 0.0, u_s.data(), 1);
            for (size_t i = 2; i < slice_points; ++i) {
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
    for (size_t ny = n_start; ny < n_end; ++ny) {
      for (size_t nz = n_start; nz < n_end; ++nz) {
        const Mesh<3> mesh{{{nx, ny, nz}},
                           Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
        const DataVector& x =
            Spectral::collocation_points(mesh.slice_through(0));
        const DataVector& y =
            Spectral::collocation_points(mesh.slice_through(1));
        const DataVector& z =
            Spectral::collocation_points(mesh.slice_through(2));
        DataVector u(mesh.number_of_grid_points());
        for (IndexIterator<3> i(mesh.extents()); i; ++i) {
          u[i.collapsed_index()] = 3 * x[i()[0]] + 5 * y[i()[1]] + z[i()[2]];
        }
        const DataVector u_lin = linearize(u, mesh);
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
    for (size_t ny = n_start; ny < n_end; ++ny) {
      for (size_t nz = n_start; nz < n_end; ++nz) {
        const Mesh<3> mesh{{{nx, ny, nz}},
                           Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
        const DataVector& x =
            Spectral::collocation_points(mesh.slice_through(0));
        const DataVector& y =
            Spectral::collocation_points(mesh.slice_through(1));
        const DataVector& z =
            Spectral::collocation_points(mesh.slice_through(2));
        DataVector u(mesh.number_of_grid_points());
        for (IndexIterator<3> i(mesh.extents()); i; ++i) {
          u[i.collapsed_index()] =
              exp(x[i()[0]]) * exp(y[i()[1]]) * exp(z[i()[2]]);
        }
        for (size_t d = 0; d < 3; ++d) {
          DataVector u_lin = linearize(u, mesh, d);
          for (StripeIterator s(mesh.extents(), d); s; ++s) {
            const Matrix& inv_v =
                Spectral::nodal_to_modal_matrix(mesh.slice_through(d));
            const auto slice_points = mesh.extents(d);
            DataVector u_s(slice_points);
            dgemv_('N', slice_points, slice_points, 1., inv_v.data(),
                   slice_points, u_lin.data() + s.offset(),  // NOLINT
                   s.stride(), 0.0, u_s.data(), 1);
            for (size_t i = 2; i < slice_points; ++i) {
              CHECK(0.0 == approx(u_s[i]));
            }
          }
        }
      }
    }
  }
}
