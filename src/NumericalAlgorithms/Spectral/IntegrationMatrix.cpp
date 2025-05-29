// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/IntegrationMatrix.hpp"

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

namespace Spectral {
template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points);

namespace {
template <Basis BasisType, Quadrature QuadratureType>
struct IntegrationMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    return Spectral::modal_to_nodal_matrix<BasisType, QuadratureType>(
               num_points) *
           spectral_indefinite_integral_matrix<BasisType>(num_points) *
           Spectral::nodal_to_modal_matrix<BasisType, QuadratureType>(
               num_points);
  }
};
}  // namespace

PRECOMPUTED_SPECTRAL_QUANTITY(integration_matrix, Matrix,
                              IntegrationMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

SPECTRAL_QUANTITY_FOR_MESH(integration_matrix, Matrix)

#undef SPECTRAL_QUANTITY_FOR_MESH

template const Matrix& integration_matrix<Basis::Chebyshev, Quadrature::Gauss>(
    size_t);
template const Matrix&
    integration_matrix<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const Matrix& integration_matrix<Basis::Legendre, Quadrature::Gauss>(
    size_t);
template const Matrix&
    integration_matrix<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
}  // namespace Spectral
