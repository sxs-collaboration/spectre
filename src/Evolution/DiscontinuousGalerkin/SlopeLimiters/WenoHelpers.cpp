// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoHelpers.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"  // IWYU pragma: keep
#include "DataStructures/Matrix.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
// Given the modal expansion corresponding to some function f(x), compute the
// modal expansion corresponding to the derivative d/dx f(x), where x are the
// logical coordinates.
//
// Note: this function assumes a Legendre basis, i.e., assumes that the modes
// are the amplitudes of a Legendre polynomial expansion.
ModalVector modal_derivative(const ModalVector& coeffs) noexcept {
  const size_t number_of_modes = coeffs.size();
  ModalVector deriv_coeffs(number_of_modes, 0.);
  for (size_t i = 0; i < number_of_modes; ++i) {
    for (size_t j = (i % 2 == 0 ? 1 : 0); j < i; j += 2) {
      deriv_coeffs[j] += coeffs[i] * (2. * j + 1.);
    }
  }
  return deriv_coeffs;
}

// For a given pair of Legendre polynomial basis functions (P_m, P_n), compute
// the sum of the integrals of all their derivatives:
// \sum_{l=0}^{N} f(l) \int d^(l)/dx^(l) P_m(x) * d^(l)/dx^(l) P_n(x) * dx
// where f(l) is some weight given to each term in the sum. In the current
// implementation we keep f(l) = 1, but this could be generalized in the future.
//
// The implemented algorithm computes the l'th derivative of P_m (and P_n) as a
// modal expansion in lower-order Legendre polynomials. The integrals can be
// carried out using the orthogonality relations between these basis functions.
double compute_sum_of_legendre_derivs(const size_t number_of_modes,
                                      const size_t m, const size_t n) noexcept {
  ModalVector coeffs_m(number_of_modes, 0.);
  coeffs_m[m] = 1.;
  ModalVector coeffs_n(number_of_modes, 0.);
  coeffs_n[n] = 1.;
  double result = 0.;
  // This is the l == 0 term of the sum
  if (m == n) {
    result += Spectral::compute_basis_function_normalization_square<
        Spectral::Basis::Legendre>(m);
  }
  // This is the main l > 0 part of the sum
  for (size_t l = 1; l < number_of_modes; ++l) {
    const double factor_of_l = 1.;
    coeffs_m = modal_derivative(coeffs_m);
    coeffs_n = modal_derivative(coeffs_n);
    for (size_t i = 0; i < number_of_modes; ++i) {
      result += factor_of_l * coeffs_m[i] * coeffs_n[i] *
                Spectral::compute_basis_function_normalization_square<
                    Spectral::Basis::Legendre>(i);
    }
  }
  return result;
}

// Compute the indicator matrix, similar to that of Dumbser2007 Eq. 25, but
// adapted for use on square/cube grids.
//
// The reference computes the indicator on a triangle/tetdrahedral grid, so
// the basis functions (and their derivatives) do not factor into a tensor
// product across dimensions. We instead compute the indicator on a square/cube
// grid, where the basis functions (and their derivatives) do factor. We use
// this factoring to write the indicator as a product of 1D contributions from
// each xi/eta/zeta component of the basis function.
//
// Note that we compute the product using derivatives 0...N in each dimension.
// We then subtract off the term with 0 derivatives in all dimensions, because
// that term is just the original data and should be omitted.
//
// Note also that because the mean value of the data does not contribute to its
// oscillation content, any matrix element associated with basis functions
// m == 0 or n == 0 vanishes. We thus slightly reduce the necessary storage by
// computing only the (N-1)^2 matrix where we start at m, n >= 1.
template <size_t VolumeDim>
Matrix compute_indicator_matrix(const Mesh<VolumeDim>& mesh) noexcept {
  ASSERT(mesh.basis() == make_array<VolumeDim>(Spectral::Basis::Legendre),
         "No implementation for mesh: " << mesh);
  Matrix result(mesh.number_of_grid_points() - 1,
                mesh.number_of_grid_points() - 1);
  for (IndexIterator<VolumeDim> m(mesh.extents()); m; ++m) {
    if (m.collapsed_index() == 0) {
      // Skip the terms associated with the cell average of the data
      continue;
    }
    for (IndexIterator<VolumeDim> n(mesh.extents()); n; ++n) {
      if (n.collapsed_index() == 0) {
        // Skip the terms associated with the cell average of the data
        continue;
      }
      // Compute each indicator matrix term by tensor-product of 1D Legendre
      // polynomial math
      auto& result_mn =
          result(m.collapsed_index() - 1, n.collapsed_index() - 1);
      result_mn =
          compute_sum_of_legendre_derivs(mesh.extents(0), m()[0], n()[0]);
      for (size_t dim = 1; dim < VolumeDim; ++dim) {
        result_mn *= compute_sum_of_legendre_derivs(mesh.extents(dim), m()[dim],
                                                    n()[dim]);
      }
    }
    // Subtract the tensor-product term that has no derivatives
    double term_with_no_derivs =
        Spectral::compute_basis_function_normalization_square<
            Spectral::Basis::Legendre>(m()[0]);
    for (size_t dim = 1; dim < VolumeDim; ++dim) {
      term_with_no_derivs *=
          Spectral::compute_basis_function_normalization_square<
              Spectral::Basis::Legendre>(m()[dim]);
    }
    result(m.collapsed_index() - 1, m.collapsed_index() - 1) -=
        term_with_no_derivs;
  }
  return result;
}

}  // namespace

namespace SlopeLimiters {
namespace Weno_detail {

template <size_t VolumeDim>
double oscillation_indicator(const DataVector& data,
                             const Mesh<VolumeDim>& mesh) noexcept {
  ASSERT(mesh.basis() == make_array<VolumeDim>(Spectral::Basis::Legendre),
         "No implementation for mesh: " << mesh);

  // Optimization: store the indicator matrix in a static so it is only computed
  // once at the beginning of the simulation. This caching will have to be
  // generalized to handle simulations with more than one mesh.
  static const Matrix indicator_matrix = compute_indicator_matrix(mesh);
  static const Mesh<VolumeDim> mesh_for_cached_matrix = mesh;
  ASSERT(
      mesh_for_cached_matrix == mesh,
      "This call to oscillation_indicator received as argument a different\n"
      "Mesh than was previously cached, suggesting that multiple meshes are\n"
      "used in the computational domain. This is not (yet) supported.\n"
      "Cached mesh: "
          << mesh_for_cached_matrix
          << "\n"
             "Argument mesh: "
          << mesh);

  const ModalVector coeffs = to_modal_coefficients(data, mesh);

  double result = 0.;
  // Note: because the 0'th modal coefficient encodes the mean of the data and
  // does not contribute to the oscillation, it is safe to exclude it from the
  // sum and start summing at m == 1, n == 1. Note also that the indicator
  // matrix is computed excluding the m == 0, n == 0 elements, so the indexing
  // is offset by 1.
  for (size_t m = 1; m < mesh.number_of_grid_points(); ++m) {
    for (size_t n = 1; n < mesh.number_of_grid_points(); ++n) {
      result += coeffs[m] * coeffs[n] * indicator_matrix(m - 1, n - 1);
    }
  }
  return result;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                        \
  template double oscillation_indicator<DIM(data)>( \
      const DataVector&, const Mesh<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Weno_detail
}  // namespace SlopeLimiters
