// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ExtremaPadding.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void check_padding(const Mesh<3>& mesh, const DataVector& grid_point_values,
                   const double true_min, const double true_max) {
  const double padding = domain::extrema_padding(mesh, grid_point_values);
  CAPTURE(padding);
  CAPTURE(min(grid_point_values));
  CAPTURE(max(grid_point_values));
  CHECK(min(grid_point_values) - padding <= true_min);
  CHECK(max(grid_point_values) + padding >= true_max);
  CHECK(padding <= 0.1 * (true_max - true_min));
}

void test_extrema_padding() {
  // Straight-sided (affine) elements: the grid points include the corners for
  // Gauss-Lobatto quadrature, so only roundoff padding is needed. For Gauss
  // quadrature the padding must cover the parts beyond the outermost points.
  for (const auto quadrature :
       {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
    CAPTURE(quadrature);
    const Mesh<3> mesh{6, Spectral::Basis::Legendre, quadrature};
    const auto logical = logical_coordinates(mesh);
    const DataVector x = 2.0 * get<0>(logical) + 0.3 * get<1>(logical) + 1.0;
    const double padding = domain::extrema_padding(mesh, x);
    if (quadrature == Spectral::Quadrature::GaussLobatto) {
      CHECK(padding < 1.0e-12);
    } else {
      check_padding(mesh, x, -1.3, 3.3);
    }
  }

  // A curved element with Legendre bases: a section of a spherical shell
  // whose outer face is centered on the x axis. The x axis is not a grid line
  // because the number of grid points is even.
  {
    const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto logical = logical_coordinates(mesh);
    const auto map = [](const double xi, const double eta, const double zeta) {
      const double r = 2.0 + 0.5 * xi;
      const double theta = 0.5 * std::numbers::pi + 0.4 * eta;
      const double phi = 0.4 * zeta;
      return std::array{r * sin(theta) * cos(phi), r * sin(theta) * sin(phi),
                        r * cos(theta)};
    };
    std::array<DataVector, 4> fields{};
    for (auto& field : fields) {
      field = DataVector{mesh.number_of_grid_points()};
    }
    for (size_t s = 0; s < mesh.number_of_grid_points(); ++s) {
      const auto x =
          map(get<0>(logical)[s], get<1>(logical)[s], get<2>(logical)[s]);
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(fields, i)[s] = gsl::at(x, i);
      }
      fields[3][s] = 2.0 + 0.5 * get<0>(logical)[s];
    }
    // Extrema over a fine sampling of the element
    std::array<double, 4> true_min{};
    std::array<double, 4> true_max{};
    true_min.fill(std::numeric_limits<double>::max());
    true_max.fill(std::numeric_limits<double>::lowest());
    const size_t n_fine = 41;
    for (size_t i = 0; i < n_fine; ++i) {
      for (size_t j = 0; j < n_fine; ++j) {
        for (size_t k = 0; k < n_fine; ++k) {
          const auto to_logical = [](const size_t index) {
            return -1.0 + 2.0 * static_cast<double>(index) /
                              static_cast<double>(n_fine - 1);
          };
          const auto x = map(to_logical(i), to_logical(j), to_logical(k));
          const std::array values{x[0], x[1], x[2], 2.0 + 0.5 * to_logical(i)};
          for (size_t c = 0; c < 4; ++c) {
            gsl::at(true_min, c) =
                std::min(gsl::at(true_min, c), gsl::at(values, c));
            gsl::at(true_max, c) =
                std::max(gsl::at(true_max, c), gsl::at(values, c));
          }
        }
      }
    }
    // The grid points miss the full extent in x
    CHECK(max(fields[0]) < true_max[0] - 1.0e-3);
    for (size_t c = 0; c < 4; ++c) {
      CAPTURE(c);
      check_padding(mesh, gsl::at(fields, c), gsl::at(true_min, c),
                    gsl::at(true_max, c));
    }
  }

  {
    const Mesh<3> mesh{
        {{6, 8, 15}},
        {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
          Spectral::Basis::SphericalHarmonic}},
        {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
          Spectral::Quadrature::Equiangular}}};
    const auto logical = logical_coordinates(mesh);
    const double offset = 0.3;
    const DataVector rho = 2.0 + 0.5 * get<0>(logical);
    const DataVector x =
        offset + rho * sin(get<1>(logical)) * cos(get<2>(logical));
    const DataVector y = rho * sin(get<1>(logical)) * sin(get<2>(logical));
    const DataVector z = rho * cos(get<1>(logical));
    const DataVector r = sqrt(square(x) + square(y) + square(z));
    CHECK(max(x) < 2.5 + offset - 1.0e-3);
    CHECK(min(x) > -2.5 + offset + 1.0e-3);
    check_padding(mesh, x, -2.5 + offset, 2.5 + offset);
    check_padding(mesh, y, -2.5, 2.5);
    check_padding(mesh, z, -2.5, 2.5);
    check_padding(mesh, r, 1.5 - offset, 2.5 + offset);
  }

  {
    const Mesh<3> mesh{{{6, 4, 9}},
                       {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                         Spectral::Basis::Fourier}},
                       {{Spectral::Quadrature::GaussLobatto,
                         Spectral::Quadrature::GaussLobatto,
                         Spectral::Quadrature::Equiangular}}};
    const auto logical = logical_coordinates(mesh);
    const DataVector rho = 2.0 + 0.5 * get<0>(logical);
    const DataVector x = rho * cos(get<2>(logical));
    CHECK(min(x) > -2.5 + 1.0e-3);
    const double padding = domain::extrema_padding(mesh, x);
    CHECK(min(x) - padding <= -2.5);
    CHECK(max(x) + padding >= 2.5);
  }
}

#ifdef SPECTRE_DEBUG
void test_errors() {
  CHECK_THROWS_WITH(
      domain::extrema_padding(Mesh<3>{4, Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto},
                              DataVector{3, 0.0}),
      Catch::Matchers::ContainsSubstring(
          "The mesh has 64 grid points but there are 3 grid point values."));
}
#endif

SPECTRE_TEST_CASE("Unit.Domain.ExtremaPadding", "[Domain][Unit]") {
  test_extrema_padding();
#ifdef SPECTRE_DEBUG
  test_errors();
#endif
}
}  // namespace
