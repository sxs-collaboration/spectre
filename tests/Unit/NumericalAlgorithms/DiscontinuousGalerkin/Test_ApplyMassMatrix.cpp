// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

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
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
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

// For a spherical-harmonic (Spherepack) angular basis the mass matrix is not a
// tensor product of per-dimension matrices (the two angular directions are
// coupled), so the `apply_matrices` reference above does not apply. Instead the
// mass matrix is diagonal in the grid index, with the angular block given by
// the Spherepack integration weights and (in 3D) the radial block by the
// standard quadrature weights, the radial index varying fastest. We build that
// diagonal directly and additionally cross-check the implied integral against
// the independent `definite_integral`.
template <typename DataType, size_t Dim>
void test_apply_mass_matrix_spherical(
    const Mesh<Dim>& mesh, const gsl::not_null<std::mt19937*> gen,
    const gsl::not_null<std::uniform_real_distribution<double>*> dist) {
  static_assert(Dim == 2 or Dim == 3);
  CAPTURE(mesh);
  const size_t num_grid_points = mesh.number_of_grid_points();
  DataVector mass_diagonal(num_grid_points);
  if constexpr (Dim == 2) {
    // Angular face: the diagonal is just the Spherepack integration weights.
    const std::vector<double>& w_angular =
        ylm::get_spherepack_cache(mesh.extents(0) - 1).integration_weights();
    for (size_t a = 0; a < num_grid_points; ++a) {
      mass_diagonal[a] = w_angular[a];
    }
  } else {
    // Volume: radial weights (dim 0) times angular weights (dims 1, 2).
    const size_t x_size = mesh.extents(0);
    const DataVector w_radial =
        Spectral::quadrature_weights(mesh.slice_through(0));
    const std::vector<double>& w_angular =
        ylm::get_spherepack_cache(mesh.extents(1) - 1).integration_weights();
    const size_t angular_size = num_grid_points / x_size;
    for (size_t a = 0; a < angular_size; ++a) {
      for (size_t i = 0; i < x_size; ++i) {
        mass_diagonal[i + x_size * a] = w_radial[i] * w_angular[a];
      }
    }
  }
  const auto field =
      make_with_random_values<DataType>(gen, dist, num_grid_points);
  CAPTURE(field);
  auto expected = field;
  for (size_t p = 0; p < num_grid_points; ++p) {
    expected[p] *= mass_diagonal[p];
  }
  {
    INFO("Test with scalar");
    auto result = field;
    apply_mass_matrix(make_not_null(&result), mesh);
    CHECK_ITERABLE_APPROX(result, expected);
    apply_inverse_mass_matrix(make_not_null(&result), mesh);
    CHECK_ITERABLE_APPROX(result, field);
  }
  {
    INFO("Test with Variables");
    using tag1 = ::Tags::TempScalar<0, DataType>;
    using tag2 = ::Tags::TempScalar<1, DataType>;
    Variables<tmpl::list<tag1, tag2>> vars{num_grid_points};
    get<tag1>(vars) = Scalar<DataType>(field);
    get<tag2>(vars) = Scalar<DataType>(field);
    apply_mass_matrix(make_not_null(&vars), mesh);
    CHECK_ITERABLE_APPROX(get(get<tag1>(vars)), expected);
    CHECK_ITERABLE_APPROX(get(get<tag2>(vars)), expected);
    apply_inverse_mass_matrix(make_not_null(&vars), mesh);
    CHECK_ITERABLE_APPROX(get(get<tag1>(vars)), field);
    CHECK_ITERABLE_APPROX(get(get<tag2>(vars)), field);
  }
  if constexpr (std::is_same_v<DataType, DataVector> and Dim == 3) {
    // Summing the mass-matrix-weighted field over the grid integrates it. This
    // compares the weight values and the radial/angular layout against the
    // independent `definite_integral` (which varies the integrand over the
    // grid, so a wrong layout would not integrate correctly).
    auto massed = field;
    apply_mass_matrix(make_not_null(&massed), mesh);
    double integral = 0.0;
    for (size_t p = 0; p < num_grid_points; ++p) {
      integral += massed[p];
    }
    CHECK(integral == approx(definite_integral(field, mesh)));
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
    {
      INFO("Spherical harmonics (Spherepack angular basis)");
      // Run both an even and an odd l_max, i.e. both odd and even n_theta grid
      for (size_t l_max = 3; l_max <= 4; ++l_max) {
        CAPTURE(l_max);
        {
          INFO("2D angular face");
          const Mesh<2> mesh{{{l_max + 1, 2 * l_max + 1}},
                             {{Spectral::Basis::SphericalHarmonic,
                               Spectral::Basis::SphericalHarmonic}},
                             {{Spectral::Quadrature::Gauss,
                               Spectral::Quadrature::Equiangular}}};
          test_apply_mass_matrix_spherical<DataType>(mesh, make_not_null(&gen),
                                                     make_not_null(&dist));
        }
        {
          INFO("3D volume");
          const Mesh<3> mesh{
              {{3, l_max + 1, 2 * l_max + 1}},
              {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
                Spectral::Basis::SphericalHarmonic}},
              {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
                Spectral::Quadrature::Equiangular}}};
          test_apply_mass_matrix_spherical<DataType>(mesh, make_not_null(&gen),
                                                     make_not_null(&dist));
        }
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
