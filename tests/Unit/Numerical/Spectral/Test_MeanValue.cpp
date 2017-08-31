// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"
#include "Numerical/Spectral/LegendreGaussLobatto.hpp"
#include "Numerical/Spectral/Linearize.hpp"
#include "Numerical/Spectral/MeanValue.hpp"
#include "tests/Unit/TestHelpers.hpp"

// GCC 4.7.3 triggers the following for the CHECK() macros.
// suggest parentheses around comparison in operand of == [-Wparentheses]
// adding an extra set of parentheses causes CHECK to print false instead
// of the lhs and rhs of the assertion when a check fails.
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wparentheses"

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.MeanValue",
                  "[Numerical][Spectral][Unit]") {
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
        DataVector u_lin = linearize(u, extents);
        int n_pts = extents.size();
        double sum = std::accumulate(u_lin.begin(), u_lin.end(), 0.0);
        CHECK(sum / n_pts == approx(mean_value(u, extents)));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.MeanValueOnBoundary",
                  "[Numerical][Spectral][Unit]") {
  for (size_t nx = 2; nx < 7; ++nx) {
    const DataVector& x = Basis::lgl::collocation_points(nx);
    for (size_t ny = 2; ny < 7; ++ny) {
      const DataVector& y = Basis::lgl::collocation_points(ny);
      for (size_t nz = 2; nz < 7; ++nz) {
        const DataVector& z = Basis::lgl::collocation_points(nz);
        const Index<3> extents(nx, ny, nz);
        DataVector u_lin = [&extents]() {
          DataVector temp(extents.product());
          return temp;
        }();
        DataVector u_quad = [&extents]() {
          DataVector temp(extents.product());
          return temp;
        }();
        for (IndexIterator<3> i(extents); i; ++i) {
          u_lin[i.offset()] = x[i()[0]] + y[i()[1]] + z[i()[2]];
          u_quad[i.offset()] = x[i()[0]] * y[i()[1]];
        }
        double mean;
        double mean_quad;
        // slice away x
        mean = mean_value_on_boundary(u_lin, extents, 0, Side::Upper);
        mean_quad = mean_value_on_boundary(u_quad, extents, 0, Side::Upper);
        CHECK(1.0 == approx(mean));
        CHECK(0.0 == approx(mean_quad));

        mean = mean_value_on_boundary(u_lin, extents, 0, Side::Lower);
        mean_quad = mean_value_on_boundary(u_quad, extents, 0, Side::Lower);
        CHECK(-1.0 == approx(mean));
        CHECK(0.0 == approx(mean_quad));

        // slice away y
        mean = mean_value_on_boundary(u_lin, extents, 1, Side::Upper);
        mean_quad = mean_value_on_boundary(u_quad, extents, 1, Side::Upper);
        CHECK(1.0 == approx(mean));
        CHECK(0.0 == approx(mean_quad));

        mean = mean_value_on_boundary(u_lin, extents, 1, Side::Lower);
        mean_quad = mean_value_on_boundary(u_quad, extents, 1, Side::Lower);
        CHECK(-1.0 == approx(mean));
        CHECK(0.0 == approx(mean_quad));

        // slice away z
        mean = mean_value_on_boundary(u_lin, extents, 2, Side::Upper);
        mean_quad = mean_value_on_boundary(u_quad, extents, 2, Side::Upper);
        CHECK(1.0 == approx(mean));
        CHECK(0.0 == approx(mean_quad));

        mean = mean_value_on_boundary(u_lin, extents, 2, Side::Lower);
        mean_quad = mean_value_on_boundary(u_quad, extents, 2, Side::Lower);
        CHECK(-1.0 == approx(mean));
        CHECK(0.0 == approx(mean_quad));
      }
    }
  }
}

//#pragma GCC diagnostic pop
