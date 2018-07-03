// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Side.hpp"
#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.MeanValue",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  constexpr size_t min_extents =
      Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>;
  constexpr size_t max_extents =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
  for (size_t nx = min_extents; nx <= max_extents; ++nx) {
    for (size_t ny = min_extents; ny <= max_extents; ++ny) {
      for (size_t nz = min_extents; nz <= max_extents; ++nz) {
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
        const DataVector u_lin = linearize(u, mesh);
        double sum = std::accumulate(u_lin.begin(), u_lin.end(), 0.0);
        CHECK(sum / mesh.number_of_grid_points() ==
              approx(mean_value(u, mesh)));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.MeanValueOnBoundary",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  constexpr size_t min_extents =
      Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>;
  constexpr size_t max_extents =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
  for (size_t nx = min_extents; nx <= max_extents; ++nx) {
    for (size_t ny = min_extents; ny <= max_extents; ++ny) {
      for (size_t nz = min_extents; nz <= max_extents; ++nz) {
        const Mesh<3> mesh{{{nx, ny, nz}},
                           Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
        const DataVector& x =
            Spectral::collocation_points(mesh.slice_through(0));
        const DataVector& y =
            Spectral::collocation_points(mesh.slice_through(1));
        const DataVector& z =
            Spectral::collocation_points(mesh.slice_through(2));
        const DataVector u_lin = [&mesh, &x, &y, &z]() {
          DataVector temp(mesh.number_of_grid_points());
          for (IndexIterator<3> i(mesh.extents()); i; ++i) {
            temp[i.collapsed_index()] = x[i()[0]] + y[i()[1]] + z[i()[2]];
          }
          return temp;
        }();
        const DataVector u_quad = [&mesh, &x, &y]() {
          DataVector temp(mesh.number_of_grid_points());
          for (IndexIterator<3> i(mesh.extents()); i; ++i) {
            temp[i.collapsed_index()] = x[i()[0]] * y[i()[1]];
          }
          return temp;
        }();
        // slice away x
        CHECK(1.0 ==
              approx(mean_value_on_boundary(u_lin, mesh, 0, Side::Upper)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, mesh, 0, Side::Upper)));

        CHECK(-1.0 ==
              approx(mean_value_on_boundary(u_lin, mesh, 0, Side::Lower)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, mesh, 0, Side::Lower)));

        // slice away y
        CHECK(1.0 ==
              approx(mean_value_on_boundary(u_lin, mesh, 1, Side::Upper)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, mesh, 1, Side::Upper)));

        CHECK(-1.0 ==
              approx(mean_value_on_boundary(u_lin, mesh, 1, Side::Lower)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, mesh, 1, Side::Lower)));

        // slice away z
        CHECK(1.0 ==
              approx(mean_value_on_boundary(u_lin, mesh, 2, Side::Upper)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, mesh, 2, Side::Upper)));

        CHECK(-1.0 ==
              approx(mean_value_on_boundary(u_lin, mesh, 2, Side::Lower)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, mesh, 2, Side::Lower)));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.MeanValueOnBoundary1D",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  constexpr size_t min_extents =
      Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>;
  constexpr size_t max_extents =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
  for (size_t nx = min_extents; nx < max_extents; ++nx) {
    const Mesh<1> mesh{nx, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const DataVector& x = Spectral::collocation_points(mesh);
    const DataVector u_lin = [&mesh, &x]() {
      DataVector temp(mesh.number_of_grid_points());
      for (IndexIterator<1> i(mesh.extents()); i; ++i) {
        temp[i.collapsed_index()] = x[i()[0]];
      }
      return temp;
    }();
    // slice away x
    CHECK(1.0 == approx(mean_value_on_boundary(u_lin, mesh, 0, Side::Upper)));
    CHECK(-1.0 == approx(mean_value_on_boundary(u_lin, mesh, 0, Side::Lower)));
  }
}
