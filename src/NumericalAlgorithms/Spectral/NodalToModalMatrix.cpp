// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"

#include <cstddef>
#include <cmath>
#include <numbers>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SpectralQuantityForMesh.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Spectral {
namespace {
template <Basis BasisType, Quadrature QuadratureType>
struct NodalToModalMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    // Numerically invert the matrix for this generic case
    return inv(modal_to_nodal_matrix<BasisType, QuadratureType>(num_points));
  }
};

template <Basis BasisType>
struct NodalToModalMatrixGenerator<BasisType, Quadrature::Gauss> {
  using data_type = Matrix;
  Matrix operator()(const size_t num_points) const {
    // For Gauss quadrature we implement the analytic expression
    // \f$\mathcal{V}^{-1}_{ij}=\mathcal{V}_{ji}\frac{w_j}{\gamma_i}\f$
    // (see description of `nodal_to_modal_matrix`).
    const DataVector& weights =
        detail::precomputed_spectral_quantity<
            BasisType, Quadrature::Gauss,
            detail::CollocationPointsAndWeightsGenerator<BasisType,
                                                         Quadrature::Gauss>>(
            num_points)
            .second;
    const Matrix& vandermonde_matrix =
        modal_to_nodal_matrix<BasisType, Quadrature::Gauss>(num_points);
    Matrix vandermonde_inverse(num_points, num_points);
    // This should be vectorized when the functionality is implemented.
    for (size_t i = 0; i < num_points; i++) {
      for (size_t j = 0; j < num_points; j++) {
        vandermonde_inverse(i, j) =
            vandermonde_matrix(j, i) * weights[j] /
            compute_basis_function_normalization_square<BasisType>(i);
      }
    }
    return vandermonde_inverse;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct TwoIndexedNodalToModalMatrixGenerator {
  Matrix operator()(const size_t num_points, const size_t m,
                    const size_t N) const {
    // To obtain the Vandermonde matrix for Zernike bases, we need to compute
    // the basis function values at the collocation points for a given angular
    // index m up to a given maximum index N.
    static_assert(BasisType == Basis::ZernikeB1 or
                  BasisType == Basis::ZernikeB2 or
                  BasisType == Basis::ZernikeB3);
    ASSERT(N < 2 * num_points - 1,
           "N must be less than 2 * num_points - 1, got N = "
               << N << " for num_points = " << num_points);
    ASSERT(m <= N, "m must be less than or equal to N, got m = "
                       << m << " and N = " << N);
    // note the integer division
    const size_t spectral_size = (N - m) / 2 + 1;
    const auto& [collocation_pts, weights] =
        compute_collocation_points_and_weights<BasisType, QuadratureType>(
            num_points);
    Matrix inv_matrix(spectral_size, num_points);
    for (size_t j = m, k = 0; j <= N; ++k, j += 2) {
      const auto& basis_function_values =
          compute_basis_function_value<BasisType>(j, m, collocation_pts);
      for (size_t i = 0; i < num_points; ++i) {
        inv_matrix(k, i) = basis_function_values[i] * weights[i];
      }
    }
    return inv_matrix;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct NodalToModalMatrixWithParityGenerator {
  Matrix operator()(const size_t num_points, const Parity parity) const {
    static_assert(BasisType == Basis::HalfFourier);
    ASSERT(parity != Parity::Uninitialized,
           "Passed parity must be set to either Even or Odd");
    const double pi_over_n = std::numbers::pi / static_cast<double>(num_points);
    const double one_over_n = 1.0 / static_cast<double>(num_points);
    const double two_over_n = 2.0 / static_cast<double>(num_points);
    Matrix inv_matrix(num_points, num_points);
    if (parity == Parity::Even) {
      for (size_t j = 0; j < num_points; ++j) {
        const double phi_j = pi_over_n * (static_cast<double>(j) + 0.5);
        inv_matrix(0, j) = one_over_n;
        for (size_t k = 1; k < num_points; ++k) {
          inv_matrix(k, j) = two_over_n * cos(static_cast<double>(k) * phi_j);
        }
      }
      return inv_matrix;
    } else {
      for (size_t j = 0; j < num_points; ++j) {
        const double phi_j = pi_over_n * (static_cast<double>(j) + 0.5);
        for (size_t k = 1; k < num_points; ++k) {
          inv_matrix(k - 1, j) =
              two_over_n * sin(static_cast<double>(k) * phi_j);
        }
        // Nyquist mode k = N stored at row index N-1
        inv_matrix(num_points - 1, j) =
            one_over_n * sin(static_cast<double>(num_points) * phi_j);
      }
    }
    return inv_matrix;
  }
};
}  // namespace

PRECOMPUTED_SPECTRAL_QUANTITY(nodal_to_modal_matrix, Matrix,
                              NodalToModalMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

SPECTRAL_QUANTITY_FOR_MESH(nodal_to_modal_matrix, Matrix)

#undef SPECTRAL_QUANTITY_FOR_MESH

PRECOMPUTED_TWO_INDEXED_SPECTRAL_QUANTITY(nodal_to_modal_matrix, Matrix,
                                          TwoIndexedNodalToModalMatrixGenerator)

#undef PRECOMPUTED_TWO_INDEXED_SPECTRAL_QUANTITY

TWO_INDEXED_SPECTRAL_QUANTITY_FOR_MESH(nodal_to_modal_matrix, Matrix)

#undef TWO_INDEXED_SPECTRAL_QUANTITY_FOR_MESH

PRECOMPUTED_SPECTRAL_QUANTITY_WITH_PARITY(nodal_to_modal_matrix, Matrix,
                                          NodalToModalMatrixWithParityGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY_WITH_PARITY

SPECTRAL_QUANTITY_WITH_PARITY_FOR_MESH(nodal_to_modal_matrix, Matrix)

#undef SPECTRAL_QUANTITY_WITH_PARITY_FOR_MESH

template const Matrix&
    nodal_to_modal_matrix<Basis::Cartoon, Quadrature::AxialSymmetry>(size_t);
template const Matrix& nodal_to_modal_matrix<
    Basis::Cartoon, Quadrature::SphericalSymmetry>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Fourier, Quadrature::Equiangular>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Legendre, Quadrature::Gauss>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
template const Matrix& nodal_to_modal_matrix<
    Basis::ZernikeB1, Quadrature::GaussRadauUpper>(size_t, size_t, size_t);
template const Matrix& nodal_to_modal_matrix<
    Basis::ZernikeB2, Quadrature::GaussRadauUpper>(size_t, size_t, size_t);
template const Matrix& nodal_to_modal_matrix<
    Basis::ZernikeB3, Quadrature::GaussRadauUpper>(size_t, size_t, size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::HalfFourier, Quadrature::Equiangular>(size_t,
                                                                       Parity);
// Some compilers require these instantiations to exist
template const Matrix& nodal_to_modal_matrix<Basis::FiniteDifference,
                                             Quadrature::CellCentered>(size_t);
template const Matrix& nodal_to_modal_matrix<Basis::FiniteDifference,
                                             Quadrature::FaceCentered>(size_t);
template const Matrix& nodal_to_modal_matrix<
    Basis::ZernikeB1, Quadrature::GaussRadauUpper>(size_t);
template const Matrix& nodal_to_modal_matrix<
    Basis::ZernikeB2, Quadrature::GaussRadauUpper>(size_t);
template const Matrix& nodal_to_modal_matrix<
    Basis::ZernikeB3, Quadrature::GaussRadauUpper>(size_t);

}  // namespace Spectral
