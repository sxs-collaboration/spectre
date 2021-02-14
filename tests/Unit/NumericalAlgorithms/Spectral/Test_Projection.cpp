// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <initializer_list>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace Spectral {
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

void test_mortar_size() {
  CHECK(get_output(Spectral::MortarSize::Full) == "Full");
  CHECK(get_output(Spectral::MortarSize::UpperHalf) == "UpperHalf");
  CHECK(get_output(Spectral::MortarSize::LowerHalf) == "LowerHalf");
}

void test_needs_projection() {
  INFO("Needs projection");
  CHECK_FALSE(needs_projection<0>({}, {}, {}));
  CHECK_FALSE(
      needs_projection<1>({3, Basis::Legendre, Quadrature::GaussLobatto},
                          {3, Basis::Legendre, Quadrature::GaussLobatto},
                          make_array<1>(ChildSize::Full)));
  CHECK_FALSE(
      needs_projection<2>({3, Basis::Legendre, Quadrature::GaussLobatto},
                          {3, Basis::Legendre, Quadrature::GaussLobatto},
                          make_array<2>(ChildSize::Full)));
  CHECK_FALSE(
      needs_projection<3>({3, Basis::Legendre, Quadrature::GaussLobatto},
                          {3, Basis::Legendre, Quadrature::GaussLobatto},
                          make_array<3>(ChildSize::Full)));
  CHECK(needs_projection<1>({3, Basis::Legendre, Quadrature::GaussLobatto},
                            {4, Basis::Legendre, Quadrature::GaussLobatto},
                            make_array<1>(ChildSize::Full)));
  CHECK(needs_projection<1>({3, Basis::Legendre, Quadrature::GaussLobatto},
                            {3, Basis::Legendre, Quadrature::Gauss},
                            make_array<1>(ChildSize::Full)));
  CHECK(needs_projection<1>({3, Basis::Legendre, Quadrature::GaussLobatto},
                            {3, Basis::Legendre, Quadrature::GaussLobatto},
                            {{ChildSize::LowerHalf}}));
  CHECK(needs_projection<2>({3, Basis::Legendre, Quadrature::GaussLobatto},
                            {3, Basis::Legendre, Quadrature::GaussLobatto},
                            {{ChildSize::Full, ChildSize::LowerHalf}}));
  CHECK(needs_projection<3>(
      {3, Basis::Legendre, Quadrature::GaussLobatto},
      {3, Basis::Legendre, Quadrature::GaussLobatto},
      {{ChildSize::Full, ChildSize::Full, ChildSize::UpperHalf}}));
}

void test_p_mortar_to_element() {
  INFO("p - mortar to element");
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
          const auto& projection = projection_matrix_child_to_parent(
              mesh_source, mesh_dest, Spectral::MortarSize::Full);
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

void test_p_element_to_mortar() {
  INFO("p - element to mortar");
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
          const auto& projection = projection_matrix_parent_to_child(
              mesh_source, mesh_dest, Spectral::MortarSize::Full);
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

  const auto& projection = projection_matrix_child_to_parent(
      mesh_self_mortar, mesh_element, mortar_size);

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

void test_h_mortar_to_element() {
  INFO("h - mortar to element");
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

void test_h_element_to_mortar() {
  INFO("h - element to mortar");
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
              const auto& projection = projection_matrix_parent_to_child(
                  mesh_source, mesh_dest, Spectral::MortarSize::UpperHalf);
              const DataVector projected_data =
                  apply_matrix(projection, source_data);
              CHECK_ITERABLE_APPROX(
                  projected_data, pow(to_upper_half(points_dest), test_order));
            }
            {
              const auto& projection = projection_matrix_parent_to_child(
                  mesh_source, mesh_dest, Spectral::MortarSize::LowerHalf);
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

void test_massive_restriction() {
  INFO("Massive restriction operator");
  // Using Gauss quadrature so the diagonal mass-matrix approximation used in
  // `::dg::apply_mass_matrix` is exact. Note that for Gauss-Lobatto quadrature
  // the mass matrix is diagonally approximated in most places in the code, but
  // the `projection_matrix_child_to_parent` uses the exact mass matrix because
  // it is implemented in terms of Vandermonde matrices.
  const Mesh<1> parent_mesh{3, Spectral::Basis::Legendre,
                            Spectral::Quadrature::Gauss};
  const Mesh<1> child_mesh{4, Spectral::Basis::Legendre,
                           Spectral::Quadrature::Gauss};
  const auto& x_child = Spectral::collocation_points(child_mesh);
  DataVector child_data = square(x_child) + x_child + 1.;
  for (const ChildSize child_size :
       {ChildSize::Full, ChildSize::LowerHalf, ChildSize::UpperHalf}) {
    CAPTURE(child_size);
    // Check R = M_coarse^-1 * I^T * M_fine and R_massive = I^T
    // => M_coarse * R * f = R_massive * M_fine * f
    //
    // (i) Compute l.h.s.
    const auto& restriction_operator =
        projection_matrix_child_to_parent(child_mesh, parent_mesh, child_size);
    auto lhs = apply_matrix(restriction_operator, child_data);
    ::dg::apply_mass_matrix(make_not_null(&lhs), parent_mesh);
    // (ii) Compute r.h.s.
    auto massive_child_data = child_data;
    if (child_size != ChildSize::Full) {
      // This is the Jacobian from logical to inertial coordinates (we take the
      // parent logical coordinates as inertial so don't have to apply a
      // Jacobian above). The `apply_mass_matrix` function requires
      // pre-multiplying by the Jacobian.
      massive_child_data *= 0.5;
    }
    ::dg::apply_mass_matrix(make_not_null(&massive_child_data), child_mesh);
    const auto& restriction_operator_massive =
        projection_matrix_child_to_parent(child_mesh, parent_mesh, child_size,
                                          true);
    const auto rhs =
        apply_matrix(restriction_operator_massive, massive_child_data);
    CHECK_ITERABLE_APPROX(lhs, rhs);
  }
}

void test_exact_restriction() {
  INFO("Exact restriction");
  const Mesh<1> child_mesh{3, Spectral::Basis::Legendre,
                           Spectral::Quadrature::Gauss};
  const Mesh<1> parent_mesh{3, Spectral::Basis::Legendre,
                            Spectral::Quadrature::Gauss};
  const auto& x_parent = Spectral::collocation_points(parent_mesh);
  const auto& xi_child = Spectral::collocation_points(child_mesh);
  const DataVector x_child_left = xi_child / 2. - 0.5;
  const DataVector x_child_right = xi_child / 2. + 0.5;
  // This polynomial is exactly represented on both the child and the parent
  // meshes
  const auto func = [](const DataVector& x) -> DataVector {
    return cube(x) + square(x) + x + 1.;
  };
  DataVector child_data_left = func(x_child_left);
  DataVector child_data_right = func(x_child_right);
  DataVector parent_data = func(x_parent);

  // Restrict function values
  const auto& restriction_operator_left = projection_matrix_child_to_parent(
      child_mesh, parent_mesh, ChildSize::LowerHalf);
  const auto& restriction_operator_right = projection_matrix_child_to_parent(
      child_mesh, parent_mesh, ChildSize::UpperHalf);
  auto restricted_data =
      apply_matrix(restriction_operator_left, child_data_left);
  restricted_data += apply_matrix(restriction_operator_right, child_data_right);
  CHECK_ITERABLE_APPROX(parent_data, restricted_data);

  // Restrict massive data
  ::dg::apply_mass_matrix(make_not_null(&parent_data), parent_mesh);
  // This is the Jacobian from logical to inertial coordinates (we take the
  // parent logical coordinates as inertial so don't have to apply a Jacobian
  // above). The `apply_mass_matrix` function requires pre-multiplying by the
  // Jacobian.
  child_data_left *= 0.5;
  child_data_right *= 0.5;
  ::dg::apply_mass_matrix(make_not_null(&child_data_left), child_mesh);
  ::dg::apply_mass_matrix(make_not_null(&child_data_right), child_mesh);
  const auto& restriction_operator_left_massive =
      projection_matrix_child_to_parent(child_mesh, parent_mesh,
                                        ChildSize::LowerHalf, true);
  const auto& restriction_operator_right_massive =
      projection_matrix_child_to_parent(child_mesh, parent_mesh,
                                        ChildSize::UpperHalf, true);
  restricted_data =
      apply_matrix(restriction_operator_left_massive, child_data_left);
  restricted_data +=
      apply_matrix(restriction_operator_right_massive, child_data_right);
  CHECK_ITERABLE_APPROX(parent_data, restricted_data);
}

template <size_t Dim>
void test_higher_dimensions() {
  INFO("Higher-dimensional operators");
  CAPTURE(Dim);
  // Higher-dimensional operators are just Cartesian products of the 1D
  // matrices, we only test here if they are constructed correctly.
  // The particular basis and quadrature don't matter for this test.
  const auto basis = Spectral::Basis::Legendre;
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  {
    INFO("Identity");
    const auto restriction_identity =
        Spectral::projection_matrix_child_to_parent(
            {3, basis, quadrature}, {3, basis, quadrature},
            make_array<Dim>(Spectral::ChildSize::Full));
    const auto prolongation_identity =
        Spectral::projection_matrix_parent_to_child(
            {3, basis, quadrature}, {3, basis, quadrature},
            make_array<Dim>(Spectral::ChildSize::Full));
    for (size_t d = 0; d < Dim;++d) {
      CHECK(gsl::at(restriction_identity, d).get() == Matrix{});
      CHECK(gsl::at(prolongation_identity, d).get() == Matrix{});
    }
  }
  {
    const size_t parent_extents = 3;
    std::array<size_t, Dim> child_extents{};
    std::iota(child_extents.begin(), child_extents.end(), size_t{3});
    auto child_sizes = make_array<Dim>(Spectral::ChildSize::Full);
    if constexpr (Dim > 1) {
      child_sizes[1] = Spectral::ChildSize::UpperHalf;
    }
    const auto projection_matrix = Spectral::projection_matrix_child_to_parent(
        {child_extents, basis, quadrature}, {parent_extents, basis, quadrature},
        child_sizes);
    CHECK(projection_matrix[0].get() == Matrix{});
    if constexpr (Dim > 1) {
      CHECK(&projection_matrix[1].get() ==
            &Spectral::projection_matrix_child_to_parent(
                {4, basis, quadrature}, {3, basis, quadrature},
                Spectral::ChildSize::UpperHalf));
    }
    if constexpr (Dim > 2) {
      CHECK(&projection_matrix[2].get() ==
            &Spectral::projection_matrix_child_to_parent(
                {5, basis, quadrature}, {3, basis, quadrature},
                Spectral::ChildSize::Full));
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Projection",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_mortar_size();
  test_needs_projection();
  test_p_mortar_to_element();
  test_p_element_to_mortar();
  test_h_mortar_to_element();
  test_h_element_to_mortar();
  test_massive_restriction();
  test_exact_restriction();
  test_higher_dimensions<1>();
  test_higher_dimensions<2>();
  test_higher_dimensions<3>();
}

}  // namespace Spectral
