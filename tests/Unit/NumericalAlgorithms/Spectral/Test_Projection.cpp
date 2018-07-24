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

namespace {
DataVector to_upper_half(const DataVector& p) noexcept {
  return 0.5 * (p + 1.);
}

DataVector to_lower_half(const DataVector& p) noexcept {
  return 0.5 * (p - 1.);
}

// `to_element_self` (`to_element_other`) is the function mapping the
// [-1,1] interval to the half that we are (are not) interpolating
// from.
template <typename F1, typename F2>
void check_mortar_to_element_projection(const Spectral::MortarSize mortar_size,
                                        const Mesh<1>& mesh_element,
                                        const Mesh<1>& mesh_self_mortar,
                                        F1&& to_element_self,
                                        F2&& to_element_other) noexcept {
  // Notation for variables in this function:
  // _self indicates the half of the interval we projected from.
  // _other indicates the half of the interval we did not project from.
  // _element indicates the coordinate system on the large interval.
  // _mortar indicates the coordinate system on one of the small intervals.

  const auto& projection = projection_matrix_mortar_to_element(
      mortar_size, mesh_element, mesh_self_mortar);

  const size_t num_points_self_mortar = mesh_self_mortar.extents(0);
  const size_t num_points_element = mesh_element.extents(0);
  const auto& points_self_mortar =
      Spectral::collocation_points(mesh_self_mortar);

  for (size_t test_order = 0;
       test_order < num_points_self_mortar;
       ++test_order) {
    CAPTURE(test_order);
    const auto test_func_self_mortar = [test_order](const auto& x) noexcept {
      return pow(x, test_order);
    };

    const DataVector data_self_mortar =
        test_func_self_mortar(points_self_mortar);
    const DataVector projected_data_element =
        apply_matrix(projection, data_self_mortar);

    // Test points for each half in each coordinate system.  These
    // have to have one extra point because LGL quadrature is not
    // sufficiently good.
    const Mesh<1> test_mesh_self_mortar(mesh_self_mortar.extents(0) + 1,
                                        mesh_self_mortar.basis(0),
                                        mesh_self_mortar.quadrature(0));
    const auto& test_points_self_mortar =
        Spectral::collocation_points(test_mesh_self_mortar);
    // We don't need to represent the initial function on the other
    // mortar.
    const Mesh<1> test_mesh_element(mesh_element.extents(0) + 1,
                                    mesh_element.basis(0),
                                    mesh_element.quadrature(0));
    const auto& test_points_other_mortar =
        Spectral::collocation_points(test_mesh_element);
    const auto& test_points_self_element =
        to_element_self(test_points_self_mortar);
    const auto& test_points_other_element =
        to_element_other(test_points_other_mortar);

    // To get the error for the half we projected from, we first
    // interpolate to the mortar at the test points, and then subtract
    // the test function at those points.
    const DataVector error_self_mortar =
        apply_matrix(Spectral::interpolation_matrix(mesh_element,
                                                    test_points_self_element),
                     projected_data_element) -
        test_func_self_mortar(test_points_self_mortar);
    // For the other half's error we can just interpolate, since the
    // source function is zero.
    const DataVector error_other_mortar = apply_matrix(
        Spectral::interpolation_matrix(mesh_element, test_points_other_element),
        projected_data_element);

    for (size_t orthogonality_test_order = 0;
         orthogonality_test_order < num_points_element;
         ++orthogonality_test_order) {
      CAPTURE(orthogonality_test_order);
      // Make sure we're using the same test function for both halves.
      // This does not have to be the same as the test function above.
      const DataVector test_function_self_mortar =
          pow(test_points_self_element, orthogonality_test_order);
      const DataVector test_function_other_mortar =
          pow(test_points_other_element, orthogonality_test_order);
      CHECK(definite_integral(error_self_mortar * test_function_self_mortar,
                              test_mesh_self_mortar) ==
            approx(-definite_integral(
                error_other_mortar * test_function_other_mortar,
                test_mesh_element)));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Projection.h.mortar_to_element",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  for (const auto& quadrature_dest : quadratures) {
    for (size_t num_points_dest = 2;
         // We need one extra point to do the quadrature later.
         num_points_dest <=
             Spectral::maximum_number_of_points<Spectral::Basis::Legendre> - 1;
         ++num_points_dest) {
      const Mesh<1> mesh_dest(num_points_dest, Spectral::Basis::Legendre,
                              quadrature_dest);
      CAPTURE(mesh_dest);
      for (const auto& quadrature_source : quadratures) {
        for (size_t num_points_source = num_points_dest;
             num_points_source <=
                 Spectral::maximum_number_of_points<Spectral::Basis::Legendre> -
                 1;
             ++num_points_source) {
          const Mesh<1> mesh_source(
              num_points_source, Spectral::Basis::Legendre, quadrature_source);
          CAPTURE(mesh_source);
          check_mortar_to_element_projection(Spectral::MortarSize::UpperHalf,
                                             mesh_dest, mesh_source,
                                             to_upper_half, to_lower_half);
          check_mortar_to_element_projection(Spectral::MortarSize::LowerHalf,
                                             mesh_dest, mesh_source,
                                             to_lower_half, to_upper_half);
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Projection.h.element_to_mortar",
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
          for (size_t test_order = 0;
               test_order < num_points_source;
               ++test_order) {
            CAPTURE(test_order);
            const DataVector source_data = pow(points_source, test_order);

            // The function is contained in the destination space, so
            // projection should not alter it.
            {
              const auto& projection = projection_matrix_element_to_mortar(
                  Spectral::MortarSize::UpperHalf, mesh_dest, mesh_source);
              const DataVector projected_data =
                  apply_matrix(projection, source_data);
              CHECK_ITERABLE_APPROX(
                  projected_data, pow(to_upper_half(points_dest), test_order));
            }
            {
              const auto& projection = projection_matrix_element_to_mortar(
                  Spectral::MortarSize::LowerHalf, mesh_dest, mesh_source);
              const DataVector projected_data =
                  apply_matrix(projection, source_data);
              CHECK_ITERABLE_APPROX(
                  projected_data, pow(to_lower_half(points_dest), test_order));
            }
          }
        }
      }
    }
  }
}
