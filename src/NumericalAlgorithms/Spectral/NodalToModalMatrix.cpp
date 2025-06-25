// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SpectralQuantityForMesh.hpp"

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
}  // namespace

PRECOMPUTED_SPECTRAL_QUANTITY(nodal_to_modal_matrix, Matrix,
                              NodalToModalMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

SPECTRAL_QUANTITY_FOR_MESH(nodal_to_modal_matrix, Matrix)

#undef SPECTRAL_QUANTITY_FOR_MESH

template const Matrix&
    nodal_to_modal_matrix<Basis::Cartoon, Quadrature::AxialSymmetry>(size_t);
template const Matrix& nodal_to_modal_matrix<
    Basis::Cartoon, Quadrature::SphericalSymmetry>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Legendre, Quadrature::Gauss>(size_t);
template const Matrix&
    nodal_to_modal_matrix<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
// Some compilers require these instantiations to exist
template const Matrix& nodal_to_modal_matrix<Basis::FiniteDifference,
                                             Quadrature::CellCentered>(size_t);
template const Matrix& nodal_to_modal_matrix<Basis::FiniteDifference,
                                             Quadrature::FaceCentered>(size_t);
}  // namespace Spectral
