// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"  // IWYU pragma: keep
#include "DataStructures/Matrix.hpp"
#include "DataStructures/ModalVector.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StaticCache.hpp"

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
// \sum_{l=0}^{N} w(l) \int d^(l)/dx^(l) P_m(x) * d^(l)/dx^(l) P_n(x) * dx
// where w(l) is some weight given to each term in the sum.
//
// The implemented algorithm computes the l'th derivative of P_m (and P_n) as a
// modal expansion in lower-order Legendre polynomials. The integrals can be
// carried out using the orthogonality relations between these basis functions.
double compute_sum_of_legendre_derivs(
    const size_t number_of_modes, const size_t m, const size_t n,
    const std::array<
        double, Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>&
        weights_for_derivatives) noexcept {
  ModalVector coeffs_m(number_of_modes, 0.);
  coeffs_m[m] = 1.;
  ModalVector coeffs_n(number_of_modes, 0.);
  coeffs_n[n] = 1.;
  double result = 0.;
  // This is the l == 0 term of the sum
  if (m == n) {
    result += weights_for_derivatives[0] *
              Spectral::compute_basis_function_normalization_square<
                  Spectral::Basis::Legendre>(m);
  }
  // This is the main l > 0 part of the sum
  for (size_t l = 1; l < number_of_modes; ++l) {
    const double weight = gsl::at(weights_for_derivatives, l);
    coeffs_m = modal_derivative(coeffs_m);
    coeffs_n = modal_derivative(coeffs_n);
    for (size_t i = 0; i < number_of_modes; ++i) {
      result += weight * coeffs_m[i] * coeffs_n[i] *
                Spectral::compute_basis_function_normalization_square<
                    Spectral::Basis::Legendre>(i);
    }
  }
  return result;
}

// Compute the indicator matrix, similar to that of Dumbser2007 Eq. 25, but
// adapted for use on square/cube grids. A Legendre basis is assumed.
//
// The reference computes the indicator on a triangle/tetrahedral grid, so
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
//
// Finally, note that the matrix measures the oscillation content in modes of
// a Legendre basis. Therefore, the matrix must be multiplied by ModalVectors
// of Legendre modes to compute the oscillation indicator.
template <size_t VolumeDim>
Matrix compute_indicator_matrix(
    const Limiters::Weno_detail::DerivativeWeight derivative_weight,
    const Index<VolumeDim>& extents) noexcept {
  Matrix result(extents.product() - 1, extents.product() - 1);

  const std::array<
      double, Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>
      weights_for_derivatives = [&derivative_weight]() noexcept {
        auto weights = make_array<
            Spectral::maximum_number_of_points<Spectral::Basis::Legendre>>(1.);
        if (derivative_weight ==
            Limiters::Weno_detail::DerivativeWeight::PowTwoEll) {
          for (size_t l = 0; l < weights.size(); ++l) {
            gsl::at(weights, l) = pow(2., 2. * l - 1.);
          }
        } else if (derivative_weight ==
                   Limiters::Weno_detail::DerivativeWeight::
                       PowTwoEllOverEllFactorial) {
          for (size_t l = 0; l < weights.size(); ++l) {
            gsl::at(weights, l) = 0.5 * square(pow(2., l) / factorial(l));
          }
        }
        return weights;
      }();

  for (IndexIterator<VolumeDim> m(extents); m; ++m) {
    if (m.collapsed_index() == 0) {
      // Skip the terms associated with the cell average of the data
      continue;
    }
    for (IndexIterator<VolumeDim> n(extents); n; ++n) {
      if (n.collapsed_index() == 0) {
        // Skip the terms associated with the cell average of the data
        continue;
      }
      // Compute each indicator matrix term by tensor-product of 1D Legendre
      // polynomial math
      auto& result_mn =
          result(m.collapsed_index() - 1, n.collapsed_index() - 1);
      result_mn = compute_sum_of_legendre_derivs(extents[0], m()[0], n()[0],
                                                 weights_for_derivatives);
      for (size_t dim = 1; dim < VolumeDim; ++dim) {
        result_mn *= compute_sum_of_legendre_derivs(
            extents[dim], m()[dim], n()[dim], weights_for_derivatives);
      }
    }
    // Subtract the tensor-product term that has no derivatives
    double term_with_no_derivs =
        weights_for_derivatives[0] *
        Spectral::compute_basis_function_normalization_square<
            Spectral::Basis::Legendre>(m()[0]);
    for (size_t dim = 1; dim < VolumeDim; ++dim) {
      term_with_no_derivs *=
          weights_for_derivatives[0] *
          Spectral::compute_basis_function_normalization_square<
              Spectral::Basis::Legendre>(m()[dim]);
    }
    result(m.collapsed_index() - 1, m.collapsed_index() - 1) -=
        term_with_no_derivs;
  }
  return result;
}

// Helper function for caching a matrix that depends on the extents of a Mesh.
// The helper handles converting from the extents (passed in as an Index) to a
// series of integers (used to call into the StaticCache) and back to an Index
// (used to compute the matrix).
template <size_t VolumeDim>
const Matrix& cached_indicator_matrix_from_mesh_index(
    const Limiters::Weno_detail::DerivativeWeight derivative_weight,
    const Index<VolumeDim>& extents) noexcept {
  using CacheEnumerationDerivativeWeight = CacheEnumeration<
      Limiters::Weno_detail::DerivativeWeight,
      Limiters::Weno_detail::DerivativeWeight::Unity,
      Limiters::Weno_detail::DerivativeWeight::PowTwoEll,
      Limiters::Weno_detail::DerivativeWeight::PowTwoEllOverEllFactorial>;
  // Oscillation indicator needs at least two grid points
  constexpr size_t min = 2;
  constexpr size_t max =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
  if constexpr (VolumeDim == 1) {
    const auto cache = make_static_cache<CacheEnumerationDerivativeWeight,
                                         CacheRange<min, max>>(
        [](const Limiters::Weno_detail::DerivativeWeight dw,
           const size_t nx) noexcept -> Matrix {
          return compute_indicator_matrix(dw, Index<1>(nx));
        });
    return cache(derivative_weight, extents[0]);
  } else if constexpr (VolumeDim == 2) {
    const auto cache =
        make_static_cache<CacheEnumerationDerivativeWeight,
                          CacheRange<min, max>, CacheRange<min, max>>(
            [](const Limiters::Weno_detail::DerivativeWeight dw,
               const size_t nx, const size_t ny) noexcept -> Matrix {
              return compute_indicator_matrix(dw, Index<2>(nx, ny));
            });
    return cache(derivative_weight, extents[0], extents[1]);
  } else {
    const auto cache =
        make_static_cache<CacheEnumerationDerivativeWeight,
                          CacheRange<min, max>, CacheRange<min, max>,
                          CacheRange<min, max>>(
            [](const Limiters::Weno_detail::DerivativeWeight dw,
               const size_t nx, const size_t ny,
               const size_t nz) noexcept -> Matrix {
              return compute_indicator_matrix(dw, Index<3>(nx, ny, nz));
            });
    return cache(derivative_weight, extents[0], extents[1], extents[2]);
  }
}

}  // namespace

namespace Limiters::Weno_detail {

std::ostream& operator<<(std::ostream& os,
                         DerivativeWeight derivative_weight) noexcept {
  switch (derivative_weight) {
    case DerivativeWeight::Unity:
      return os << "Unity";
    case DerivativeWeight::PowTwoEll:
      return os << "PowTwoEll";
    case DerivativeWeight::PowTwoEllOverEllFactorial:
      return os << "PowTwoEllOverEllFactorial";
    default:
      ERROR("Unknown DerivativeWeight");
  }
}

template <size_t VolumeDim>
double oscillation_indicator(const DerivativeWeight derivative_weight,
                             const DataVector& data,
                             const Mesh<VolumeDim>& mesh) noexcept {
  ASSERT(mesh.basis() == make_array<VolumeDim>(Spectral::Basis::Legendre),
         "Unsupported basis: " << mesh);
  ASSERT(mesh.quadrature() ==
                 make_array<VolumeDim>(Spectral::Quadrature::GaussLobatto) or
             mesh.quadrature() ==
                 make_array<VolumeDim>(Spectral::Quadrature::Gauss),
         "Unsupported quadrature: " << mesh);
  // The oscillation indicator is computed from the N>0 spectral modes of the
  // input data, so we need at least two modes => at least two grid points.
  ASSERT(*alg::min_element(mesh.extents().indices()) > 1,
         "Unsupported extents: " << mesh);

  const Matrix& indicator_matrix = cached_indicator_matrix_from_mesh_index(
      derivative_weight, mesh.extents());
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
      DerivativeWeight, const DataVector&, const Mesh<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Limiters::Weno_detail
