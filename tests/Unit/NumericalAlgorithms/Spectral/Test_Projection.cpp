// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <initializer_list>
#include <random>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/FourierTestFunctions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

namespace Spectral {
namespace {
constexpr auto quadratures = {Spectral::Quadrature::Gauss,
                              Spectral::Quadrature::GaussLobatto};

DataVector apply_matrix(const Matrix& m, const DataVector& v) {
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
  CHECK(get_output(Spectral::SegmentSize::Full) == "Full");
  CHECK(get_output(Spectral::SegmentSize::UpperHalf) == "UpperHalf");
  CHECK(get_output(Spectral::SegmentSize::LowerHalf) == "LowerHalf");
}

void test_needs_projection() {
  INFO("Needs projection");
  CHECK_FALSE(needs_projection<0>({}, {}, {}));
  CHECK_FALSE(
      needs_projection<1>({3, Basis::Legendre, Quadrature::GaussLobatto},
                          {3, Basis::Legendre, Quadrature::GaussLobatto},
                          make_array<1>(SegmentSize::Full)));
  CHECK_FALSE(
      needs_projection<2>({3, Basis::Legendre, Quadrature::GaussLobatto},
                          {3, Basis::Legendre, Quadrature::GaussLobatto},
                          make_array<2>(SegmentSize::Full)));
  CHECK_FALSE(
      needs_projection<3>({3, Basis::Legendre, Quadrature::GaussLobatto},
                          {3, Basis::Legendre, Quadrature::GaussLobatto},
                          make_array<3>(SegmentSize::Full)));
  CHECK(needs_projection<1>({3, Basis::Legendre, Quadrature::GaussLobatto},
                            {4, Basis::Legendre, Quadrature::GaussLobatto},
                            make_array<1>(SegmentSize::Full)));
  CHECK(needs_projection<1>({3, Basis::Legendre, Quadrature::GaussLobatto},
                            {3, Basis::Legendre, Quadrature::Gauss},
                            make_array<1>(SegmentSize::Full)));
  CHECK(needs_projection<1>({3, Basis::Legendre, Quadrature::GaussLobatto},
                            {3, Basis::Legendre, Quadrature::GaussLobatto},
                            {{SegmentSize::LowerHalf}}));
  CHECK(needs_projection<2>({3, Basis::Legendre, Quadrature::GaussLobatto},
                            {3, Basis::Legendre, Quadrature::GaussLobatto},
                            {{SegmentSize::Full, SegmentSize::LowerHalf}}));
  CHECK(needs_projection<3>(
      {3, Basis::Legendre, Quadrature::GaussLobatto},
      {3, Basis::Legendre, Quadrature::GaussLobatto},
      {{SegmentSize::Full, SegmentSize::Full, SegmentSize::UpperHalf}}));
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
      const auto& points_dest = Spectral::collocation_points(mesh_dest);
      CAPTURE(mesh_dest);
      for (const auto& quadrature_source : quadratures) {
        for (size_t num_points_source = 2;
             num_points_source <=
             Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
             ++num_points_source) {
          const Mesh<1> mesh_source(
              num_points_source, Spectral::Basis::Legendre, quadrature_source);
          CAPTURE(mesh_source);
          const auto& points_source = Spectral::collocation_points(mesh_source);
          const auto& projection = projection_matrix_child_to_parent(
              mesh_source, mesh_dest, Spectral::SegmentSize::Full);
          if (num_points_source <= num_points_dest) {
            const auto& parent_to_child_projection =
                projection_matrix_parent_to_child(mesh_source, mesh_dest,
                                                  Spectral::SegmentSize::Full);
            CHECK(projection == parent_to_child_projection);
          }
          for (size_t test_order = 0; test_order < num_points_source;
               ++test_order) {
            CAPTURE(test_order);
            const DataVector source_data = pow(points_source, test_order);
            const DataVector projected_data =
                apply_matrix(projection, source_data);
            if (num_points_source > num_points_dest) {
              // Projection matrices can be defined as the matrices which
              // make the error in the destination space orthogonal to the
              // destination space.  We interpolate back to the higher
              // dimensional source space to check.
              const DataVector interpolated_projected_data = apply_matrix(
                  Spectral::interpolation_matrix(mesh_dest, points_source),
                  projected_data);
              const DataVector error =
                  interpolated_projected_data - source_data;

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
            } else {
              // The function can be represented exactly in both spaces.
              auto local_approx = Approx::custom().scale(1.0).epsilon(1.0e-13);
              CHECK_ITERABLE_CUSTOM_APPROX(
                  projected_data, pow(points_dest, test_order), local_approx);
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
        for (size_t num_points_source = 2; num_points_source <= num_points_dest;
             ++num_points_source) {
          const Mesh<1> mesh_source(
              num_points_source, Spectral::Basis::Legendre, quadrature_source);
          CAPTURE(mesh_source);
          const auto& points_source = Spectral::collocation_points(mesh_source);
          const auto& projection = projection_matrix_parent_to_child(
              mesh_source, mesh_dest, Spectral::SegmentSize::Full);
          for (size_t test_order = 0; test_order < num_points_source;
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

DataVector to_upper_half(const DataVector& p) { return 0.5 * (p + 1.); }

DataVector to_lower_half(const DataVector& p) { return 0.5 * (p - 1.); }

// `to_element_self` (`to_element_other`) is the function mapping the
// [-1,1] interval to the half that we are (are not) interpolating
// from.
template <typename F1, typename F2>
void check_mortar_to_element_projection(const Spectral::SegmentSize mortar_size,
                                        const Mesh<1>& mesh_element,
                                        const Mesh<1>& mesh_self_mortar,
                                        F1&& to_element_self,
                                        F2&& to_element_other) {
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

  for (size_t test_order = 0; test_order < num_points_self_mortar;
       ++test_order) {
    CAPTURE(test_order);
    const auto test_func_self_mortar = [test_order](const auto& x) {
      return pow(x, test_order);
    };

    const DataVector data_self_mortar =
        test_func_self_mortar(points_self_mortar);
    const DataVector projected_data_element =
        apply_matrix(projection, data_self_mortar);

    // Test points for each half in each coordinate system.  These
    // have to have one extra point because LGL quadrature is not
    // sufficiently good.
    const Mesh<1> test_mesh_self_mortar(
        std::max(mesh_self_mortar.extents(0), mesh_element.extents(0)) + 1,
        mesh_self_mortar.basis(0), mesh_self_mortar.quadrature(0));
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
        for (size_t num_points_source = 2;
             num_points_source <=
             Spectral::maximum_number_of_points<Spectral::Basis::Legendre> - 1;
             ++num_points_source) {
          const Mesh<1> mesh_source(
              num_points_source, Spectral::Basis::Legendre, quadrature_source);
          CAPTURE(mesh_source);
          check_mortar_to_element_projection(Spectral::SegmentSize::UpperHalf,
                                             mesh_dest, mesh_source,
                                             to_upper_half, to_lower_half);
          check_mortar_to_element_projection(Spectral::SegmentSize::LowerHalf,
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
        for (size_t num_points_source = 2; num_points_source <= num_points_dest;
             ++num_points_source) {
          const Mesh<1> mesh_source(
              num_points_source, Spectral::Basis::Legendre, quadrature_source);
          CAPTURE(mesh_source);
          const auto& points_source = Spectral::collocation_points(mesh_source);
          for (size_t test_order = 0; test_order < num_points_source;
               ++test_order) {
            CAPTURE(test_order);
            const DataVector source_data = pow(points_source, test_order);

            // The function is contained in the destination space, so
            // projection should not alter it.
            {
              const auto& projection = projection_matrix_parent_to_child(
                  mesh_source, mesh_dest, Spectral::SegmentSize::UpperHalf);
              const DataVector projected_data =
                  apply_matrix(projection, source_data);
              CHECK_ITERABLE_APPROX(
                  projected_data, pow(to_upper_half(points_dest), test_order));
            }
            {
              const auto& projection = projection_matrix_parent_to_child(
                  mesh_source, mesh_dest, Spectral::SegmentSize::LowerHalf);
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

void test_massive_restriction(const size_t parent_num_points,
                              const size_t child_num_points) {
  INFO("Massive restriction operator");
  CAPTURE(parent_num_points);
  CAPTURE(child_num_points);
  REQUIRE(parent_num_points < child_num_points);
  // Using Gauss quadrature so the diagonal mass-matrix approximation used in
  // `::dg::apply_mass_matrix` is exact. Note that for Gauss-Lobatto quadrature
  // the mass matrix is diagonally approximated in most places in the code, but
  // the `projection_matrix_child_to_parent` uses the exact mass matrix because
  // it is implemented in terms of Vandermonde matrices.
  const Mesh<1> parent_mesh{parent_num_points, Spectral::Basis::Legendre,
                            Spectral::Quadrature::Gauss};
  const Mesh<1> child_mesh{child_num_points, Spectral::Basis::Legendre,
                           Spectral::Quadrature::Gauss};
  const auto& x_child = Spectral::collocation_points(child_mesh);
  DataVector child_data = square(x_child) + x_child + 1.;
  for (const SegmentSize child_size :
       {SegmentSize::Full, SegmentSize::LowerHalf, SegmentSize::UpperHalf}) {
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
    if (child_size != SegmentSize::Full) {
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
    Approx local_approx = Approx::custom().epsilon(1.0e-9);
    CHECK_ITERABLE_CUSTOM_APPROX(lhs, rhs, local_approx);
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
      child_mesh, parent_mesh, SegmentSize::LowerHalf);
  const auto& restriction_operator_right = projection_matrix_child_to_parent(
      child_mesh, parent_mesh, SegmentSize::UpperHalf);
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
                                        SegmentSize::LowerHalf, true);
  const auto& restriction_operator_right_massive =
      projection_matrix_child_to_parent(child_mesh, parent_mesh,
                                        SegmentSize::UpperHalf, true);
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
            make_array<Dim>(Spectral::SegmentSize::Full));
    const auto prolongation_identity =
        Spectral::projection_matrix_parent_to_child(
            {3, basis, quadrature}, {3, basis, quadrature},
            make_array<Dim>(Spectral::SegmentSize::Full));
    for (size_t d = 0; d < Dim; ++d) {
      CHECK(gsl::at(restriction_identity, d).get() == Matrix{});
      CHECK(gsl::at(prolongation_identity, d).get() == Matrix{});
    }
  }
  {
    const size_t parent_extents = 3;
    std::array<size_t, Dim> child_extents{};
    std::iota(child_extents.begin(), child_extents.end(), size_t{3});
    auto child_sizes = make_array<Dim>(Spectral::SegmentSize::Full);
    if constexpr (Dim > 1) {
      child_sizes[1] = Spectral::SegmentSize::UpperHalf;
    }
    const auto projection_matrix = Spectral::projection_matrix_child_to_parent(
        {child_extents, basis, quadrature}, {parent_extents, basis, quadrature},
        child_sizes);
    CHECK(projection_matrix[0].get() == Matrix{});
    if constexpr (Dim > 1) {
      CHECK(&projection_matrix[1].get() ==
            &Spectral::projection_matrix_child_to_parent(
                {4, basis, quadrature}, {3, basis, quadrature},
                Spectral::SegmentSize::UpperHalf));
    }
    if constexpr (Dim > 2) {
      CHECK(&projection_matrix[2].get() ==
            &Spectral::projection_matrix_child_to_parent(
                {5, basis, quadrature}, {3, basis, quadrature},
                Spectral::SegmentSize::Full));
    }
  }
}

template <size_t Dim>
void test_p_projection_matrices() {
  INFO("p-projection operators");
  CAPTURE(Dim);
  // Higher-dimensional operators are just Cartesian products of the 1D
  // matrices, we only test here if they are constructed correctly.
  // The particular basis and quadrature don't matter for this test.
  const auto basis = Spectral::Basis::Legendre;
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  {
    INFO("Identity");
    const auto identity = Spectral::projection_matrices(
        Mesh<Dim>{3, basis, quadrature}, Mesh<Dim>{3, basis, quadrature},
        make_array<Dim>(SegmentSize::Full), make_array<Dim>(SegmentSize::Full));
    for (size_t d = 0; d < Dim; ++d) {
      CHECK(gsl::at(identity, d).get() == Matrix{});
    }
  }
  {
    const size_t source_extents = 4;
    std::array<size_t, Dim> target_extents{};
    std::iota(target_extents.begin(), target_extents.end(), size_t{3});
    const auto projection_matrix = Spectral::projection_matrices(
        Mesh<Dim>{source_extents, basis, quadrature},
        Mesh<Dim>{target_extents, basis, quadrature},
        make_array<Dim>(SegmentSize::Full), make_array<Dim>(SegmentSize::Full));
    CHECK(&projection_matrix[0].get() ==
          &Spectral::projection_matrix_child_to_parent(
              {4, basis, quadrature}, {3, basis, quadrature},
              Spectral::SegmentSize::Full));
    if constexpr (Dim > 1) {
      CHECK(projection_matrix[1].get() == Matrix{});
    }
    if constexpr (Dim > 2) {
      CHECK(&projection_matrix[2].get() ==
            &Spectral::projection_matrix_child_to_parent(
                {4, basis, quadrature}, {5, basis, quadrature},
                Spectral::SegmentSize::Full));
      // Interpolation is the same as mode-padding
      CHECK(projection_matrix[2].get() ==
            Spectral::projection_matrix_parent_to_child(
                {4, basis, quadrature}, {5, basis, quadrature},
                Spectral::SegmentSize::Full));
    }
  }
}

template <size_t Dim>
void test_projection_matrices() {
  INFO("generic projection operators");
  CAPTURE(Dim);
  // Higher-dimensional operators are just Cartesian products of the 1D
  // matrices, we only test here if they are constructed correctly.
  // The particular basis and quadrature don't matter for this test.
  const auto basis = Spectral::Basis::Legendre;
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  {
    INFO("Identity");
    const auto identity = Spectral::projection_matrices(
        Mesh<Dim>{3, basis, quadrature}, Mesh<Dim>{3, basis, quadrature},
        make_array<Dim>(SegmentSize::Full), make_array<Dim>(SegmentSize::Full));
    for (size_t d = 0; d < Dim; ++d) {
      CHECK(gsl::at(identity, d).get() == Matrix{});
    }
  }
  {
    const size_t source_extents = 4;
    std::array<size_t, Dim> target_extents{};
    std::iota(target_extents.begin(), target_extents.end(), size_t{3});
    std::array<SegmentSize, Dim> source_sizes{};
    std::array<SegmentSize, Dim> target_sizes{};
    source_sizes[0] = SegmentSize::LowerHalf;
    target_sizes[0] = SegmentSize::LowerHalf;
    if constexpr (Dim > 1) {
      source_sizes[1] = SegmentSize::Full;
      target_sizes[1] = SegmentSize::UpperHalf;
    }
    if constexpr (Dim > 2) {
      source_sizes[2] = SegmentSize::LowerHalf;
      target_sizes[2] = SegmentSize::Full;
    }
    const auto projection_matrix = Spectral::projection_matrices(
        Mesh<Dim>{source_extents, basis, quadrature},
        Mesh<Dim>{target_extents, basis, quadrature}, source_sizes,
        target_sizes);
    CHECK(&projection_matrix[0].get() ==
          &Spectral::projection_matrix_child_to_parent(
              {4, basis, quadrature}, {3, basis, quadrature},
              Spectral::SegmentSize::Full));
    if constexpr (Dim > 1) {
      CHECK(&projection_matrix[1].get() ==
            &Spectral::projection_matrix_parent_to_child(
                {4, basis, quadrature}, {4, basis, quadrature},
                Spectral::SegmentSize::UpperHalf));
    }
    if constexpr (Dim > 2) {
      CHECK(&projection_matrix[2].get() ==
            &Spectral::projection_matrix_child_to_parent(
                {4, basis, quadrature}, {5, basis, quadrature},
                Spectral::SegmentSize::LowerHalf));
    }
  }
}

void test_fourier_p_projections() {
  // Test both child_to_parent and parent_to_child projections for Fourier basis
  // For Fourier basis, SegmentSize::Full only--Fourier does not support
  // h-refinement
  INFO("Fourier p-projections");
  const auto local_approx = Approx::custom().scale(1.0).epsilon(1.0e-14);
  constexpr size_t max_points_f =
      Spectral::maximum_number_of_points<Spectral::Basis::Fourier>;
  const std::vector<std::pair<unsigned, unsigned>> test_funcs = {
      {1_st, 0_st}, {0_st, 1_st}, {1_st, 1_st}, {5_st, 4_st}, {13_st, 15_st}};

  // source <= dest (prolongation)
  constexpr size_t stride = 5;
  for (size_t num_points_source = 2; num_points_source <= max_points_f;
       num_points_source += stride) {
    const Mesh<1> mesh_source(num_points_source, Spectral::Basis::Fourier,
                              Spectral::Quadrature::Equiangular);
    const auto& points_source = Spectral::collocation_points(mesh_source);
    CAPTURE(mesh_source);
    for (size_t num_points_dest = num_points_source + stride;
         num_points_dest <= std::min(3 * num_points_source, max_points_f);
         num_points_dest += stride) {
      const Mesh<1> mesh_dest(num_points_dest, Spectral::Basis::Fourier,
                              Spectral::Quadrature::Equiangular);
      const auto& points_dest = Spectral::collocation_points(mesh_dest);
      CAPTURE(mesh_dest);

      // Test child_to_parent (mortar to element)
      const auto& projection_child_to_parent =
          projection_matrix_child_to_parent(mesh_source, mesh_dest,
                                            Spectral::SegmentSize::Full);
      // Test parent_to_child (element to mortar)
      const auto& projection_parent_to_child =
          projection_matrix_parent_to_child(mesh_source, mesh_dest,
                                            Spectral::SegmentSize::Full);

      // For Fourier with SegmentSize::Full, these should be the same matrix
      CHECK(&projection_child_to_parent == &projection_parent_to_child);

      for (const auto& [pow_nx, pow_ny] : test_funcs) {
        const size_t required_modes = pow_nx + pow_ny;
        if (required_modes >= num_points_source / 2) {
          continue;
        }
        CAPTURE(pow_nx);
        CAPTURE(pow_ny);
        const FourierTestFunctions::ProductOfPolynomials func(pow_nx, pow_ny);

        const DataVector projected_data_ctp =
            apply_matrix(projection_child_to_parent, func(points_source));
        CHECK_ITERABLE_CUSTOM_APPROX(projected_data_ctp, func(points_dest),
                                     local_approx);
      }
    }
  }

  // source > dest (restriction)
  for (size_t num_points_dest = 2; num_points_dest <= max_points_f;
       num_points_dest += stride) {
    const Mesh<1> mesh_dest(num_points_dest, Spectral::Basis::Fourier,
                            Spectral::Quadrature::Equiangular);
    CAPTURE(mesh_dest);
    for (size_t num_points_source = num_points_dest + stride;
         num_points_source <= std::min(3 * num_points_dest, max_points_f);
         num_points_source += stride) {
      const Mesh<1> mesh_source(num_points_source, Spectral::Basis::Fourier,
                                Spectral::Quadrature::Equiangular);
      const auto& points_source = Spectral::collocation_points(mesh_source);
      CAPTURE(mesh_source);

      // Test both projection functions
      const auto& projection_child_to_parent =
          projection_matrix_child_to_parent(mesh_source, mesh_dest,
                                            Spectral::SegmentSize::Full);
      const auto& projection_parent_to_child =
          projection_matrix_parent_to_child(mesh_source, mesh_dest,
                                            Spectral::SegmentSize::Full);

      // For Fourier with SegmentSize::Full, these should be the same matrix
      CHECK(&projection_child_to_parent == &projection_parent_to_child);

      for (const auto& [pow_nx, pow_ny] : test_funcs) {
        const size_t required_modes = pow_nx + pow_ny;
        if (required_modes >= num_points_source / 2) {
          continue;
        }
        CAPTURE(pow_nx);
        CAPTURE(pow_ny);
        const FourierTestFunctions::ProductOfPolynomials func(pow_nx, pow_ny);
        const DataVector projected_data =
            apply_matrix(projection_child_to_parent, func(points_source));
        // Interpolate projected result back to the fine grid and measure the
        // truncation error.
        const DataVector interpolated_projected_data = apply_matrix(
            Spectral::interpolation_matrix(mesh_dest, points_source),
            projected_data);
        const DataVector error =
            interpolated_projected_data - func(points_source);
        // The error must be L2-orthogonal to every basis function that fits
        // in the destination mesh
        for (size_t index = 0; index < num_points_dest; ++index) {
          CAPTURE(index);
          const DataVector basis_func_values =
              Spectral::compute_basis_function_value<Spectral::Basis::Fourier>(
                  index, points_source);
          CHECK(definite_integral(error * basis_func_values, mesh_source) ==
                local_approx(0.0));
        }
      }
    }
  }
}

void test_zernike_b2_fourier_equivalence() {
  INFO("ZernikeB2-Fourier equivalence");
  // Test that ZernikeB2 with Equiangular behaves identically to Fourier
  const size_t parent_size = 5;
  const size_t child_size = 9;
  const Mesh<1> fourier_parent(parent_size, Spectral::Basis::Fourier,
                               Spectral::Quadrature::Equiangular);
  const Mesh<1> fourier_child(child_size, Spectral::Basis::Fourier,
                              Spectral::Quadrature::Equiangular);
  const Mesh<1> zernike_parent(parent_size, Spectral::Basis::ZernikeB2,
                               Spectral::Quadrature::Equiangular);
  const Mesh<1> zernike_child(child_size, Spectral::Basis::ZernikeB2,
                              Spectral::Quadrature::Equiangular);

  const auto& f_to_f_child_to_parent = projection_matrix_child_to_parent(
      fourier_child, fourier_parent, Spectral::SegmentSize::Full);
  const auto& f_to_z_child_to_parent = projection_matrix_child_to_parent(
      fourier_child, zernike_parent, Spectral::SegmentSize::Full);
  CHECK(&f_to_f_child_to_parent == &f_to_z_child_to_parent);
  const auto& z_to_z_child_to_parent = projection_matrix_child_to_parent(
      zernike_child, zernike_parent, Spectral::SegmentSize::Full);
  CHECK(&f_to_f_child_to_parent == &z_to_z_child_to_parent);

  const auto& f_to_f_parent_to_child = projection_matrix_child_to_parent(
      fourier_parent, fourier_child, Spectral::SegmentSize::Full);
  const auto& f_to_z_parent_to_child = projection_matrix_child_to_parent(
      fourier_parent, zernike_child, Spectral::SegmentSize::Full);
  CHECK(&f_to_f_parent_to_child == &f_to_z_parent_to_child);
  const auto& z_to_z_parent_to_child = projection_matrix_child_to_parent(
      zernike_parent, zernike_child, Spectral::SegmentSize::Full);
  CHECK(&f_to_f_parent_to_child == &z_to_z_parent_to_child);
}

#ifdef SPECTRE_DEBUG
void test_fourier_and_zernikeb2_asserts() {
  {
    INFO("Fourier h-refinement asserts");
    const Mesh<1> mesh(5, Spectral::Basis::Fourier,
                       Spectral::Quadrature::Equiangular);
    CHECK_THROWS_WITH(projection_matrix_child_to_parent(
                          mesh, mesh, Spectral::SegmentSize::UpperHalf),
                      Catch::Matchers::ContainsSubstring(
                          "Unsupported projection combination"));
    CHECK_THROWS_WITH(projection_matrix_child_to_parent(
                          mesh, mesh, Spectral::SegmentSize::LowerHalf),
                      Catch::Matchers::ContainsSubstring(
                          "Unsupported projection combination"));
  }
  {
    INFO("ZernikeB2 projection asserts");

    // Test that ZernikeB2 without Equiangular quadrature is rejected
    const Mesh<1> zernike_gauss{4, Basis::ZernikeB2, Quadrature::Gauss};
    const Mesh<1> fourier_mesh{4, Basis::Fourier, Quadrature::Equiangular};

    CHECK_THROWS_WITH(projection_matrix_child_to_parent(
                          zernike_gauss, fourier_mesh, SegmentSize::Full),
                      Catch::Matchers::ContainsSubstring(
                          "Unsupported projection combination"));

    // Test that ZernikeB2 with h-refinement is rejected
    const Mesh<1> zernike_equi{4, Basis::ZernikeB2, Quadrature::Equiangular};

    CHECK_THROWS_WITH(projection_matrix_child_to_parent(
                          zernike_equi, zernike_equi, SegmentSize::UpperHalf),
                      Catch::Matchers::ContainsSubstring(
                          "Unsupported projection combination"));

    CHECK_THROWS_WITH(projection_matrix_child_to_parent(
                          zernike_equi, zernike_equi, SegmentSize::LowerHalf),
                      Catch::Matchers::ContainsSubstring(
                          "Unsupported projection combination"));
  }
}
#endif  // SPECTRE_DEBUG

void test_hash() {
  CHECK(Spectral::MortarSizeHash<1>{}({}) == 0);

  CHECK(Spectral::MortarSizeHash<2>{}({Spectral::SegmentSize::LowerHalf}) == 0);
  CHECK(Spectral::MortarSizeHash<2>{}({Spectral::SegmentSize::Full}) == 0);
  CHECK(Spectral::MortarSizeHash<2>{}({Spectral::SegmentSize::UpperHalf}) == 1);

  CHECK(Spectral::MortarSizeHash<3>{}({Spectral::SegmentSize::LowerHalf,
                                       Spectral::SegmentSize::LowerHalf}) == 0);
  CHECK(Spectral::MortarSizeHash<3>{}({Spectral::SegmentSize::UpperHalf,
                                       Spectral::SegmentSize::LowerHalf}) == 1);
  CHECK(Spectral::MortarSizeHash<3>{}({Spectral::SegmentSize::Full,
                                       Spectral::SegmentSize::LowerHalf}) == 0);

  CHECK(Spectral::MortarSizeHash<3>{}({Spectral::SegmentSize::LowerHalf,
                                       Spectral::SegmentSize::UpperHalf}) == 2);
  CHECK(Spectral::MortarSizeHash<3>{}({Spectral::SegmentSize::UpperHalf,
                                       Spectral::SegmentSize::UpperHalf}) == 3);
  CHECK(Spectral::MortarSizeHash<3>{}({Spectral::SegmentSize::Full,
                                       Spectral::SegmentSize::UpperHalf}) == 2);

  CHECK(Spectral::MortarSizeHash<3>{}({Spectral::SegmentSize::LowerHalf,
                                       Spectral::SegmentSize::Full}) == 0);
  CHECK(Spectral::MortarSizeHash<3>{}({Spectral::SegmentSize::UpperHalf,
                                       Spectral::SegmentSize::Full}) == 1);
  CHECK(Spectral::MortarSizeHash<3>{}(
            {Spectral::SegmentSize::Full, Spectral::SegmentSize::Full}) == 0);
}

// Tags used by the Variables overload test of Spectral::project.
struct Scalar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Scalar2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

Mesh<3> shell_mesh(const size_t num_radial_points, const size_t l_max) {
  return Mesh<3>{
      {{num_radial_points, l_max + 1, 2 * l_max + 1}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
}

Mesh<2> sphere_mesh(const size_t l_max) {
  return Mesh<2>{
      {{l_max + 1, 2 * l_max + 1}},
      {{Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}}};
}

// Build volume data on a spherical shell representing the separable function
// radial_poly(r) * angular(theta, phi), where the angular part is defined by
// `reference_modes` (the Spherepack spectral coefficients at `l_reference`,
// with no content above that degree) and radial_poly is the polynomial with the
// given `radial_coeffs` in the logical radial coordinate. As long as the mesh
// resolves both factors (l_max >= l_reference and num_radial_points > degree),
// this is the *same* function on every such mesh, so Spectral::project between
// two of them must reproduce it exactly.
DataVector shell_field(const Mesh<3>& mesh, const DataVector& reference_modes,
                       const size_t l_reference,
                       const std::vector<double>& radial_coeffs) {
  const size_t l_max = mesh.extents(1) - 1;
  const size_t num_radial_points = mesh.extents(0);
  const ylm::Spherepack& ylm = ylm::get_spherepack_cache(l_max);
  const DataVector modes = ylm::Spherepack::prolong_or_restrict(
      reference_modes, l_reference, l_reference, l_max, l_max);
  const DataVector angular = ylm.spec_to_phys(modes);
  const DataVector radial = evaluate_polynomial(
      radial_coeffs, Spectral::collocation_points(mesh.slice_through(0)));
  DataVector field(num_radial_points * angular.size());
  for (size_t a = 0; a < angular.size(); ++a) {
    for (size_t r = 0; r < num_radial_points; ++r) {
      field[r + num_radial_points * a] = radial[r] * angular[a];
    }
  }
  return field;
}

// The angular factor of shell_field on its own, as a function on the 2D sphere
// `mesh`. It contains no angular modes above degree `l_reference`, so it is the
// same function on every mesh that resolves it (l_max >= l_reference) and
// Spectral::project between two such meshes must reproduce it exactly.
DataVector sphere_field(const Mesh<2>& mesh, const DataVector& reference_modes,
                        const size_t l_reference) {
  const size_t l_max = mesh.extents(0) - 1;
  const ylm::Spherepack& ylm = ylm::get_spherepack_cache(l_max);
  const DataVector modes = ylm::Spherepack::prolong_or_restrict(
      reference_modes, l_reference, l_reference, l_max, l_max);
  return ylm.spec_to_phys(modes);
}

// The cube (tensor-product) path of Spectral::project must reproduce exactly
// the behavior of the matrix-returning API, so existing consumers are
// unchanged.
void test_project_cube() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  // p-refinement
  {
    const Mesh<2> source{{{3, 4}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const Mesh<2> target{{{5, 6}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    DataVector data(source.number_of_grid_points());
    fill_with_random_values(make_not_null(&data), make_not_null(&gen),
                            make_not_null(&dist));
    DataVector result{};
    Spectral::project(make_not_null(&result), data, source, target,
                      make_array<2>(Spectral::SegmentSize::Full),
                      make_array<2>(Spectral::SegmentSize::Full));
    const DataVector expected = apply_matrices(
        Spectral::projection_matrices(
            source, target, make_array<2>(Spectral::SegmentSize::Full),
            make_array<2>(Spectral::SegmentSize::Full)),
        data, source.extents());
    CHECK_ITERABLE_APPROX(result, expected);
  }
  // h-refinement (parent -> child) in dimension 0
  {
    const Mesh<2> parent{{{4, 4}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const Mesh<2> child{{{4, 4}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
    const std::array child_sizes{Spectral::SegmentSize::UpperHalf,
                                 Spectral::SegmentSize::Full};
    DataVector data(parent.number_of_grid_points());
    fill_with_random_values(make_not_null(&data), make_not_null(&gen),
                            make_not_null(&dist));
    DataVector result{};
    Spectral::project(make_not_null(&result), data, parent, child,
                      make_array<2>(Spectral::SegmentSize::Full), child_sizes);
    const DataVector expected = apply_matrices(
        Spectral::projection_matrix_parent_to_child(parent, child, child_sizes),
        data, parent.extents());
    CHECK_ITERABLE_APPROX(result, expected);
  }
  // massive restriction (child -> parent)
  {
    const Mesh<2> child{{{5, 5}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
    const Mesh<2> parent{{{3, 3}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const std::array child_sizes{Spectral::SegmentSize::LowerHalf,
                                 Spectral::SegmentSize::Full};
    DataVector data(child.number_of_grid_points());
    fill_with_random_values(make_not_null(&data), make_not_null(&gen),
                            make_not_null(&dist));
    DataVector result{};
    Spectral::project(make_not_null(&result), data, child, parent, child_sizes,
                      make_array<2>(Spectral::SegmentSize::Full), true);
    const DataVector expected =
        apply_matrices(Spectral::projection_matrix_child_to_parent(
                           child, parent, child_sizes, true),
                       data, child.extents());
    CHECK_ITERABLE_APPROX(result, expected);
  }
}

void test_project_shell() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const size_t l_reference = 4;
  const ylm::Spherepack& ylm_reference = ylm::get_spherepack_cache(l_reference);
  DataVector reference_phys(ylm_reference.physical_size());
  fill_with_random_values(make_not_null(&reference_phys), make_not_null(&gen),
                          make_not_null(&dist));
  const DataVector reference_modes = ylm_reference.phys_to_spec(reference_phys);
  std::vector<double> radial_coeffs(4);  // polynomial of degree 3
  fill_with_random_values(make_not_null(&radial_coeffs), make_not_null(&gen),
                          make_not_null(&dist));

  const auto check_exact = [&reference_modes, &radial_coeffs](
                               const Mesh<3>& source, const Mesh<3>& target) {
    const DataVector source_field =
        shell_field(source, reference_modes, l_reference, radial_coeffs);
    const DataVector target_field =
        shell_field(target, reference_modes, l_reference, radial_coeffs);
    DataVector result{};
    Spectral::project(make_not_null(&result), source_field, source, target,
                      make_array<3>(Spectral::SegmentSize::Full),
                      make_array<3>(Spectral::SegmentSize::Full));
    CHECK_ITERABLE_APPROX(result, target_field);
  };

  const auto coarse = shell_mesh(5, 6);
  const auto fine = shell_mesh(7, 10);
  // identity
  check_exact(coarse, coarse);
  // combined radial + angular, both directions
  check_exact(coarse, fine);
  check_exact(fine, coarse);
  // radial-only (angular l_max unchanged)
  check_exact(shell_mesh(5, 6), shell_mesh(8, 6));
  check_exact(shell_mesh(8, 6), shell_mesh(5, 6));
  // angular-only (radial unchanged)
  check_exact(shell_mesh(5, 6), shell_mesh(5, 9));
  check_exact(shell_mesh(5, 9), shell_mesh(5, 6));

  // prolong-then-restrict round trip returns the original coarse field
  {
    const DataVector coarse_field =
        shell_field(coarse, reference_modes, l_reference, radial_coeffs);
    DataVector fine_result{};
    Spectral::project(make_not_null(&fine_result), coarse_field, coarse, fine,
                      make_array<3>(Spectral::SegmentSize::Full),
                      make_array<3>(Spectral::SegmentSize::Full));
    DataVector round_trip{};
    Spectral::project(make_not_null(&round_trip), fine_result, fine, coarse,
                      make_array<3>(Spectral::SegmentSize::Full),
                      make_array<3>(Spectral::SegmentSize::Full));
    CHECK_ITERABLE_APPROX(round_trip, coarse_field);
  }

  // Variables overload is consistent with the DataVector overload
  {
    std::vector<double> other_radial_coeffs(4);
    fill_with_random_values(make_not_null(&other_radial_coeffs),
                            make_not_null(&gen), make_not_null(&dist));
    using vars_tags = tmpl::list<Scalar1, Scalar2>;
    Variables<vars_tags> source_vars(coarse.number_of_grid_points());
    get(get<Scalar1>(source_vars)) =
        shell_field(coarse, reference_modes, l_reference, radial_coeffs);
    get(get<Scalar2>(source_vars)) =
        shell_field(coarse, reference_modes, l_reference, other_radial_coeffs);
    Variables<vars_tags> result_vars{};
    Spectral::project(make_not_null(&result_vars), source_vars, coarse, fine,
                      make_array<3>(Spectral::SegmentSize::Full),
                      make_array<3>(Spectral::SegmentSize::Full));
    DataVector expected1{};
    DataVector expected2{};
    Spectral::project(make_not_null(&expected1), get(get<Scalar1>(source_vars)),
                      coarse, fine, make_array<3>(Spectral::SegmentSize::Full),
                      make_array<3>(Spectral::SegmentSize::Full));
    Spectral::project(make_not_null(&expected2), get(get<Scalar2>(source_vars)),
                      coarse, fine, make_array<3>(Spectral::SegmentSize::Full),
                      make_array<3>(Spectral::SegmentSize::Full));
    CHECK_ITERABLE_APPROX(get(get<Scalar1>(result_vars)), expected1);
    CHECK_ITERABLE_APPROX(get(get<Scalar2>(result_vars)), expected2);
  }
}

// 2D spherical shell (sphere surface): angular-only projection, exercised
// through both the DataVector and Variables overloads.
void test_project_shell_2d() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const size_t l_reference = 4;
  const ylm::Spherepack& ylm_reference = ylm::get_spherepack_cache(l_reference);
  DataVector reference_phys(ylm_reference.physical_size());
  fill_with_random_values(make_not_null(&reference_phys), make_not_null(&gen),
                          make_not_null(&dist));
  const DataVector reference_modes = ylm_reference.phys_to_spec(reference_phys);

  const auto coarse = sphere_mesh(6);
  const auto fine = sphere_mesh(10);
  const auto full = make_array<2>(Spectral::SegmentSize::Full);
  const auto check_exact = [&reference_modes, &full](const Mesh<2>& source,
                                                     const Mesh<2>& target) {
    const DataVector source_field =
        sphere_field(source, reference_modes, l_reference);
    const DataVector target_field =
        sphere_field(target, reference_modes, l_reference);
    DataVector result{};
    Spectral::project(make_not_null(&result), source_field, source, target,
                      full, full);
    CHECK_ITERABLE_APPROX(result, target_field);
  };
  check_exact(coarse, coarse);  // identity
  check_exact(coarse, fine);
  check_exact(fine, coarse);

  // Variables overload is consistent with the DataVector overload
  using vars_tags = tmpl::list<Scalar1, Scalar2>;
  Variables<vars_tags> source_vars(coarse.number_of_grid_points());
  get(get<Scalar1>(source_vars)) =
      sphere_field(coarse, reference_modes, l_reference);
  get(get<Scalar2>(source_vars)) =
      2.0 * sphere_field(coarse, reference_modes, l_reference);
  Variables<vars_tags> result_vars{};
  Spectral::project(make_not_null(&result_vars), source_vars, coarse, fine,
                    full, full);
  CHECK_ITERABLE_APPROX(get(get<Scalar1>(result_vars)),
                        sphere_field(fine, reference_modes, l_reference));
  CHECK_ITERABLE_APPROX(get(get<Scalar2>(result_vars)),
                        2.0 * sphere_field(fine, reference_modes, l_reference));
}

// The massive restriction must be the transpose of the (non-massive)
// prolongation, just as for tensor-product meshes. Equivalently, for the
// Euclidean inner products,
//   <u_coarse, R_massive v_fine> == <I u_coarse, v_fine>,
// where I is the prolongation coarse -> fine. This adjoint property is what
// geometric multigrid relies on, so verify it across radial-only, angular-only,
// and combined refinement of a spherical shell.
void test_project_shell_massive() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto full = make_array<3>(Spectral::SegmentSize::Full);
  const auto custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  const auto check_adjoint = [&](const Mesh<3>& coarse, const Mesh<3>& fine) {
    DataVector u_coarse(coarse.number_of_grid_points());
    DataVector v_fine(fine.number_of_grid_points());
    fill_with_random_values(make_not_null(&u_coarse), make_not_null(&gen),
                            make_not_null(&dist));
    fill_with_random_values(make_not_null(&v_fine), make_not_null(&gen),
                            make_not_null(&dist));
    DataVector prolonged{};
    Spectral::project(make_not_null(&prolonged), u_coarse, coarse, fine, full,
                      full);
    DataVector restricted{};
    Spectral::project(make_not_null(&restricted), v_fine, fine, coarse, full,
                      full, true);
    CHECK(sum(u_coarse * restricted) == custom_approx(sum(prolonged * v_fine)));
  };
  check_adjoint(shell_mesh(4, 4), shell_mesh(6, 7));  // radial and angular
  check_adjoint(shell_mesh(4, 5), shell_mesh(6, 5));  // radial only
  check_adjoint(shell_mesh(5, 4), shell_mesh(5, 7));  // angular only
}

#ifdef SPECTRE_DEBUG
void test_project_shell_asserts() {
  const auto coarse = shell_mesh(5, 6);
  const auto fine = shell_mesh(7, 10);
  DataVector data(coarse.number_of_grid_points(), 1.0);
  DataVector result{};
  // angular dimensions cannot be h-refined
  CHECK_THROWS_WITH(
      Spectral::project(make_not_null(&result), data, coarse, fine,
                        std::array{Spectral::SegmentSize::Full,
                                   Spectral::SegmentSize::UpperHalf,
                                   Spectral::SegmentSize::Full},
                        make_array<3>(Spectral::SegmentSize::Full)),
      Catch::Matchers::ContainsSubstring("angular h-refinement"));
}
#endif  // SPECTRE_DEBUG
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Projection",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_mortar_size();
  test_needs_projection();
  test_p_mortar_to_element();
  test_p_element_to_mortar();
  test_h_mortar_to_element();
  test_h_element_to_mortar();
  for (size_t child_num_points = 4;
       child_num_points <= maximum_number_of_points<Spectral::Basis::Legendre>;
       ++child_num_points) {
    for (size_t parent_num_points = 3; parent_num_points < child_num_points;
         ++parent_num_points) {
      test_massive_restriction(parent_num_points, child_num_points);
    }
  }
  test_exact_restriction();
  test_higher_dimensions<1>();
  test_higher_dimensions<2>();
  test_higher_dimensions<3>();
  test_p_projection_matrices<1>();
  test_p_projection_matrices<2>();
  test_p_projection_matrices<3>();
  test_projection_matrices<1>();
  test_projection_matrices<2>();
  test_projection_matrices<3>();
  test_fourier_p_projections();
  test_zernike_b2_fourier_equivalence();
  test_project_cube();
  test_project_shell();
  test_project_shell_2d();
  test_project_shell_massive();
#ifdef SPECTRE_DEBUG
  test_fourier_and_zernikeb2_asserts();
  test_project_shell_asserts();
#endif  // SPECTRE_DEBUG
  test_hash();
}

}  // namespace Spectral
