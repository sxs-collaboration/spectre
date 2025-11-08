// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/LinearFilteringMatrix.hpp"

#include <cstddef>

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SpectralQuantityForMesh.hpp"
#include "Utilities/Blas.hpp"

namespace Spectral {
namespace {

template <Basis BasisType, Quadrature QuadratureType>
struct LinearFilterMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    // We implement the expression
    // \f$\mathcal{V}^{-1}\cdot\mathrm{diag}(1,1,0,0,...)\cdot\mathcal{V}\f$
    // (see description of `linear_filter_matrix`)
    // which multiplies the first two columns of
    // `nodal_to_modal_matrix` with the first two rows of
    // `modal_to_nodal_matrix`.
    Matrix lin_filter(num_points, num_points);
    dgemm_(
        'N', 'N', num_points, num_points, std::min(size_t{2}, num_points), 1.0,
        modal_to_nodal_matrix<BasisType, QuadratureType>(num_points).data(),
        modal_to_nodal_matrix<BasisType, QuadratureType>(num_points).spacing(),
        nodal_to_modal_matrix<BasisType, QuadratureType>(num_points).data(),
        nodal_to_modal_matrix<BasisType, QuadratureType>(num_points).spacing(),
        0.0, lin_filter.data(), lin_filter.spacing());
    return lin_filter;
  }
};
}  // namespace

PRECOMPUTED_SPECTRAL_QUANTITY(linear_filter_matrix, Matrix,
                              LinearFilterMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

SPECTRAL_QUANTITY_FOR_MESH(linear_filter_matrix, Matrix)

#undef SPECTRAL_QUANTITY_FOR_MESH

template const Matrix&
    linear_filter_matrix<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const Matrix&
    linear_filter_matrix<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const Matrix& linear_filter_matrix<Basis::Legendre, Quadrature::Gauss>(
    size_t);
template const Matrix&
    linear_filter_matrix<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
template const Matrix&
linear_filter_matrix<Basis::Fourier, Quadrature::Equiangular>(size_t);
}  // namespace Spectral
