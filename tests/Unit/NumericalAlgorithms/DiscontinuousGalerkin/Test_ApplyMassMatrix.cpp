// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace {

// This function computes the logical mass matrix exactly using the (normalized)
// Vandermonde matrix: M = (V * V^T)^-1. We don't currently use this way of
// computing the mass matrix in the DG code because it is cheaper to apply the
// mass matrix as a pointwise multiplication over the grid.
template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
Matrix exact_logical_mass_matrix(const size_t num_points) {
  auto normalized_vandermonde_matrix =
      Spectral::modal_to_nodal_matrix<BasisType, QuadratureType>(num_points);
  for (size_t j = 0; j < num_points; ++j) {
    const double normalization = sqrt(
        Spectral::compute_basis_function_normalization_square<BasisType>(j));
    for (size_t i = 0; i < num_points; ++i) {
      normalized_vandermonde_matrix(i, j) /= normalization;
    }
  }
  return inv(normalized_vandermonde_matrix *
             trans(normalized_vandermonde_matrix));
}

template <size_t Dim>
std::array<Matrix, Dim> exact_logical_mass_matrix(const Mesh<Dim>& mesh) {
  std::array<Matrix, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT(mesh.basis(d) == Spectral::Basis::Legendre,
           "This function is currently only implemented for a Legendre basis.");
    switch (mesh.quadrature(d)) {
      case Spectral::Quadrature::Gauss: {
        gsl::at(result, d) =
            exact_logical_mass_matrix<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss>(
                mesh.extents(d));
        break;
      }
      case Spectral::Quadrature::GaussLobatto: {
        gsl::at(result, d) =
            exact_logical_mass_matrix<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto>(
                mesh.extents(d));
        break;
      }
      default:
        ERROR(
            "This function is currently only implemented for Gauss and "
            "Gauss-Lobatto quadrature.");
    }
  }
  return result;
}

template <size_t Dim>
std::array<Matrix, Dim> diag_logical_mass_matrix(const Mesh<Dim>& mesh) {
  std::array<Matrix, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    const size_t num_points = mesh.extents(d);
    const auto& weights = Spectral::quadrature_weights(mesh.slice_through(d));
    gsl::at(result, d) = Matrix(num_points, num_points, 0.);
    for (size_t i = 0; i < num_points; ++i) {
      gsl::at(result, d)(i, i) = weights[i];
    }
  }
  return result;
}

template <typename DataType, size_t Dim>
void test_apply_mass_matrix(
    const Mesh<Dim>& mesh, const std::array<Matrix, Dim>& mass_matrix,
    const gsl::not_null<std::mt19937*> gen,
    const gsl::not_null<std::uniform_real_distribution<double>*> dist) {
  CAPTURE(mesh);
  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto scalar_field =
      make_with_random_values<DataType>(gen, dist, num_grid_points);
  CAPTURE(scalar_field);
  const auto expected_mass_matrix_times_scalar_field =
      apply_matrices(mass_matrix, scalar_field, mesh.extents());
  {
    INFO("Test with scalar");
    auto result = scalar_field;
    apply_mass_matrix(make_not_null(&result), mesh);
    CHECK_ITERABLE_APPROX(result, expected_mass_matrix_times_scalar_field);
    apply_inverse_mass_matrix(make_not_null(&result), mesh);
    CHECK_ITERABLE_APPROX(result, scalar_field);
  }
  {
    INFO("Test with Variables");
    using tag1 = ::Tags::TempScalar<0, DataType>;
    using tag2 = ::Tags::TempScalar<1, DataType>;
    Variables<tmpl::list<tag1, tag2>> vars{num_grid_points};
    get<tag1>(vars) = Scalar<DataType>(scalar_field);
    get<tag2>(vars) = Scalar<DataType>(scalar_field);
    apply_mass_matrix(make_not_null(&vars), mesh);
    CHECK_ITERABLE_APPROX(get(get<tag1>(vars)),
                          expected_mass_matrix_times_scalar_field);
    CHECK_ITERABLE_APPROX(get(get<tag2>(vars)),
                          expected_mass_matrix_times_scalar_field);
    apply_inverse_mass_matrix(make_not_null(&vars), mesh);
    CHECK_ITERABLE_APPROX(get(get<tag1>(vars)), scalar_field);
    CHECK_ITERABLE_APPROX(get(get<tag2>(vars)), scalar_field);
  }
}

template <typename DataType>
void test_apply_mass_matrix() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(-1., 1.);
  {
    INFO("1D");
    {
      INFO("Gauss quadrature (exact)");
      const Mesh<1> mesh{
          {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
      test_apply_mass_matrix<DataType>(mesh, exact_logical_mass_matrix(mesh),
                                       make_not_null(&gen),
                                       make_not_null(&dist));
    }
    {
      INFO("Gauss-Lobatto quadrature");
      const Mesh<1> mesh{
          {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
      test_apply_mass_matrix<DataType>(mesh, diag_logical_mass_matrix(mesh),
                                       make_not_null(&gen),
                                       make_not_null(&dist));
    }
    {
      INFO("2D");
      {
        INFO("Gauss quadrature (exact)");
        const Mesh<2> mesh{
            {{4, 2}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
        test_apply_mass_matrix<DataType>(mesh, exact_logical_mass_matrix(mesh),
                                         make_not_null(&gen),
                                         make_not_null(&dist));
      }
      {
        INFO("Gauss-Lobatto quadrature");
        const Mesh<2> mesh{{{4, 2}},
                           Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
        test_apply_mass_matrix<DataType>(mesh, diag_logical_mass_matrix(mesh),
                                         make_not_null(&gen),
                                         make_not_null(&dist));
      }
    }
    {
      INFO("3D");
      {
        INFO("Gauss quadrature (exact)");
        const Mesh<3> mesh{{{4, 2, 3}},
                           Spectral::Basis::Legendre,
                           Spectral::Quadrature::Gauss};
        test_apply_mass_matrix<DataType>(mesh, exact_logical_mass_matrix(mesh),
                                         make_not_null(&gen),
                                         make_not_null(&dist));
      }
      {
        INFO("Gauss-Lobatto quadrature");
        const Mesh<3> mesh{{{4, 2, 3}},
                           Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
        test_apply_mass_matrix<DataType>(mesh, diag_logical_mass_matrix(mesh),
                                         make_not_null(&gen),
                                         make_not_null(&dist));
      }
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.DiscontinuousGalerkin.ApplyMassMatrix",
                  "[NumericalAlgorithms][Unit]") {
  test_apply_mass_matrix<DataVector>();
  test_apply_mass_matrix<ComplexDataVector>();
}

}  // namespace dg
