// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.MeanValue",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  for (size_t nx = 2; nx < 7; ++nx) {
    const DataVector& x = Basis::lgl::collocation_points(nx);
    for (size_t ny = 2; ny < 7; ++ny) {
      const DataVector& y = Basis::lgl::collocation_points(ny);
      for (size_t nz = 2; nz < 7; ++nz) {
        const DataVector& z = Basis::lgl::collocation_points(nz);
        const Index<3> extents(nx, ny, nz);
        DataVector u(extents.product());
        for (IndexIterator<3> i(extents); i; ++i) {
          u[i.offset()] = exp(x[i()[0]]) * exp(y[i()[1]]) * exp(z[i()[2]]);
        }
        const DataVector u_lin = linearize(u, extents);
        size_t n_pts = extents.product();
        double sum = std::accumulate(u_lin.begin(), u_lin.end(), 0.0);
        CHECK(sum / n_pts == approx(mean_value(u, extents)));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.MeanValueOnBoundary",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  for (size_t nx = 2; nx < 7; ++nx) {
    const DataVector& x = Basis::lgl::collocation_points(nx);
    for (size_t ny = 2; ny < 7; ++ny) {
      const DataVector& y = Basis::lgl::collocation_points(ny);
      for (size_t nz = 2; nz < 7; ++nz) {
        const DataVector& z = Basis::lgl::collocation_points(nz);
        const Index<3> extents(nx, ny, nz);
        const DataVector u_lin = [&extents, &x, &y, &z]() {
          DataVector temp(extents.product());
          for (IndexIterator<3> i(extents); i; ++i) {
            temp[i.offset()] = x[i()[0]] + y[i()[1]] + z[i()[2]];
          }
          return temp;
        }();
        const DataVector u_quad = [&extents, &x, &y]() {
          DataVector temp(extents.product());
          for (IndexIterator<3> i(extents); i; ++i) {
            temp[i.offset()] = x[i()[0]] * y[i()[1]];
          }
          return temp;
        }();
        // slice away x
        CHECK(1.0 ==
              approx(mean_value_on_boundary(u_lin, extents, 0, Side::Upper)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, extents, 0, Side::Upper)));

        CHECK(-1.0 ==
              approx(mean_value_on_boundary(u_lin, extents, 0, Side::Lower)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, extents, 0, Side::Lower)));

        // slice away y
        CHECK(1.0 ==
              approx(mean_value_on_boundary(u_lin, extents, 1, Side::Upper)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, extents, 1, Side::Upper)));

        CHECK(-1.0 ==
              approx(mean_value_on_boundary(u_lin, extents, 1, Side::Lower)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, extents, 1, Side::Lower)));

        // slice away z
        CHECK(1.0 ==
              approx(mean_value_on_boundary(u_lin, extents, 2, Side::Upper)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, extents, 2, Side::Upper)));

        CHECK(-1.0 ==
              approx(mean_value_on_boundary(u_lin, extents, 2, Side::Lower)));
        CHECK(0.0 ==
              approx(mean_value_on_boundary(u_quad, extents, 2, Side::Lower)));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.MeanValueOnBoundary1D",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  for (size_t nx = 2; nx < 7; ++nx) {
    const DataVector& x = Basis::lgl::collocation_points(nx);
    const Index<1> extents(nx);
    const DataVector u_lin = [&extents, &x]() {
      DataVector temp(extents.product());
      for (IndexIterator<1> i(extents); i; ++i) {
        temp[i.offset()] = x[i()[0]];
      }
      return temp;
    }();
    // slice away x
    CHECK(1.0 ==
          approx(mean_value_on_boundary(u_lin, extents, 0, Side::Upper)));
    CHECK(-1.0 ==
          approx(mean_value_on_boundary(u_lin, extents, 0, Side::Lower)));
  }
}
