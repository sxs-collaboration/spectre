// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
Matrix exact_logical_mass_matrix(const size_t num_points) noexcept {
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
std::array<Matrix, Dim> exact_logical_mass_matrix(
    const Mesh<Dim>& mesh) noexcept {
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
void test_apply_mass_matrix(
    const Mesh<Dim>& mesh, const DataVector& scalar_field,
    const DataVector& expected_mass_matrix_times_scalar_field) noexcept {
  const size_t num_grid_points = mesh.number_of_grid_points();
  {
    INFO("Test with DataVector");
    auto result = scalar_field;
    apply_mass_matrix(make_not_null(&result), mesh);
    CHECK_ITERABLE_APPROX(result, expected_mass_matrix_times_scalar_field);
  }
  {
    INFO("Test with Variables");
    using tag1 = ::Tags::TempScalar<0>;
    using tag2 = ::Tags::TempScalar<1>;
    Variables<tmpl::list<tag1, tag2>> vars{num_grid_points};
    get<tag1>(vars) = Scalar<DataVector>(scalar_field);
    get<tag2>(vars) = Scalar<DataVector>(scalar_field);
    apply_mass_matrix(make_not_null(&vars), mesh);
    CHECK_ITERABLE_APPROX(get(get<tag1>(vars)),
                          expected_mass_matrix_times_scalar_field);
    CHECK_ITERABLE_APPROX(get(get<tag2>(vars)),
                          expected_mass_matrix_times_scalar_field);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.DiscontinuousGalerkin.ApplyMassMatrix",
                  "[NumericalAlgorithms][Unit]") {
  {
    INFO("1D");
    {
      INFO("Gauss quadrature (exact)");
      const Mesh<1> mesh{
          {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
      const DataVector data{1., 2., 3., 4.};
      const auto mass_matrix = exact_logical_mass_matrix(mesh);
      const auto massive_data =
          apply_matrices(mass_matrix, data, mesh.extents());
      test_apply_mass_matrix(mesh, data, massive_data);
    }
    {
      INFO("Gauss-Lobatto quadrature");
      const Mesh<1> mesh{
          {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
      test_apply_mass_matrix(mesh, {1., 2., 3., 4.},
                             // Data divided by LGL quadrature weights
                             DataVector{1., 10., 15., 4.} / 6.);
    }
  }
  {
    INFO("2D");
    {
      INFO("Gauss quadrature (exact)");
      const Mesh<2> mesh{
          {{4, 2}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
      const DataVector data{1., 2., 3., 4., 5., 6., 7., 8.};
      const auto mass_matrix = exact_logical_mass_matrix(mesh);
      const auto massive_data =
          apply_matrices(mass_matrix, data, mesh.extents());
      test_apply_mass_matrix(mesh, data, massive_data);
    }
    {
      INFO("Gauss-Lobatto quadrature");
      const Mesh<2> mesh{{{4, 2}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
      test_apply_mass_matrix(
          mesh, {1., 2., 3., 4., 5., 6., 7., 8.},
          // Data divided by LGL quadrature weights
          DataVector{1., 10., 15., 4., 5., 30., 35., 8.} / 6.);
    }
  }
  {
    INFO("3D");
    {
      INFO("Gauss quadrature (exact)");
      const Mesh<3> mesh{
          {{4, 2, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
      const DataVector data{1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                            9.,  10., 11., 12., 13., 14., 15., 16.,
                            17., 18., 19., 20., 21., 22., 23., 24.};
      const auto mass_matrix = exact_logical_mass_matrix(mesh);
      const auto massive_data =
          apply_matrices(mass_matrix, data, mesh.extents());
      test_apply_mass_matrix(mesh, data, massive_data);
    }
    {
      INFO("Gauss-Lobatto quadrature");
      const Mesh<3> mesh{{{4, 2, 3}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
      test_apply_mass_matrix(
          mesh, {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.},
          // Data divided by LGL quadrature weights
          DataVector{1.,  10.,  15.,  4.,  5.,  30.,  35.,  8.,
                     36., 200., 220., 48., 52., 280., 300., 64.,
                     17., 90.,  95.,  20., 21., 110., 115., 24.} /
              18.);
    }
  }
}

}  // namespace dg
