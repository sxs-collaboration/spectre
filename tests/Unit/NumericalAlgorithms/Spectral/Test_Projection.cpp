// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <initializer_list>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
constexpr auto quadratures = {Spectral::Quadrature::Gauss,
                              Spectral::Quadrature::GaussLobatto};

DataVector apply_matrix(const Matrix& m, const DataVector& v) noexcept {
  ASSERT(m.columns() == v.size(), "Bad apply_matrix");
  DataVector result(m.rows(), 0.);
  for (size_t i = 0; i < m.rows(); ++i) {
    for (size_t j = 0; j < m.columns(); ++j) {
      result[i] += m(i, j) * v[j];
    }
  }
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Projection.p.mortar_to_element",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  for (const auto& quadrature_dest : quadratures) {
    for (size_t num_points_dest = 2;
         num_points_dest <=
             Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
         ++num_points_dest) {
      const Mesh<1> mesh_dest(num_points_dest, Spectral::Basis::Legendre,
                              quadrature_dest);
      CAPTURE(mesh_dest);
      for (const auto& quadrature_source : quadratures) {
        for (size_t num_points_source = num_points_dest;
             num_points_source <=
                 Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
             ++num_points_source) {
          const Mesh<1> mesh_source(
              num_points_source, Spectral::Basis::Legendre, quadrature_source);
          CAPTURE(mesh_source);
          const auto& points_source = Spectral::collocation_points(mesh_source);
          const auto& projection = projection_matrix_mortar_to_element(
              Spectral::MortarSize::Full, mesh_dest, mesh_source);
          for (size_t test_order = 0;
               test_order < num_points_source;
               ++test_order) {
            CAPTURE(test_order);
            const DataVector source_data = pow(points_source, test_order);
            const DataVector projected_data =
                apply_matrix(projection, source_data);
            // Projection matrices can be defined as the matrices which
            // make the error in the destination space orthogonal to the
            // destination space.  We interpolate back to the higher
            // dimensional source space to check.
            const DataVector interpolated_projected_data = apply_matrix(
                Spectral::interpolation_matrix(mesh_dest, points_source),
                projected_data);
            const DataVector error = interpolated_projected_data - source_data;

            for (size_t orthogonality_test_order = 0;
                 orthogonality_test_order < num_points_dest;
                 ++orthogonality_test_order) {
              // This integral might not be evaluated exactly for the
              // highest order polynomials, but it will correctly
              // determine orthogonality.
              CHECK(definite_integral(
                        error * pow(points_source, orthogonality_test_order),
                        mesh_source) == approx(0.));
            }
          }
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Projection.p.element_to_mortar",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  for (const auto& quadrature_dest : quadratures) {
    for (size_t num_points_dest = 2;
         num_points_dest <=
             Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
         ++num_points_dest) {
      const Mesh<1> mesh_dest(num_points_dest, Spectral::Basis::Legendre,
                              quadrature_dest);
      CAPTURE(mesh_dest);
      const auto& points_dest = Spectral::collocation_points(mesh_dest);
      for (const auto& quadrature_source : quadratures) {
        for (size_t num_points_source = 2;
             num_points_source <= num_points_dest;
             ++num_points_source) {
          const Mesh<1> mesh_source(
              num_points_source, Spectral::Basis::Legendre, quadrature_source);
          CAPTURE(mesh_source);
          const auto& points_source = Spectral::collocation_points(mesh_source);
          const auto& projection = projection_matrix_element_to_mortar(
              Spectral::MortarSize::Full, mesh_dest, mesh_source);
          for (size_t test_order = 0;
               test_order < num_points_source;
               ++test_order) {
            CAPTURE(test_order);
            const DataVector source_data = pow(points_source, test_order);
            const DataVector projected_data =
                apply_matrix(projection, source_data);
            // The function is contained in the destination space, so
            // projection should not alter it.
            CHECK_ITERABLE_APPROX(projected_data, pow(points_dest, test_order));
          }
        }
      }
    }
  }
}
