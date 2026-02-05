// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SpectralQuantityForMesh.hpp"

namespace Spectral {
namespace {
template <Basis BasisType, Quadrature QuadratureType>
struct ModalToNodalMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    ASSERT(BasisType != Basis::ZernikeB1 and BasisType != Basis::ZernikeB2 and
               BasisType != Basis::ZernikeB3,
           "Zernike basis should not call one-indexed ModalToNodal matrix");
    // To obtain the Vandermonde matrix we need to compute the basis function
    // values at the collocation points. Constructing the matrix proceeds
    // the same for any basis.
    const DataVector& collocation_pts =
        collocation_points<BasisType, QuadratureType>(num_points);
    Matrix vandermonde_matrix(num_points, num_points);
    for (size_t j = 0; j < num_points; j++) {
      const auto& basis_function_values =
          compute_basis_function_value<BasisType>(j, collocation_pts);
      for (size_t i = 0; i < num_points; i++) {
        vandermonde_matrix(i, j) = basis_function_values[i];
      }
    }
    return vandermonde_matrix;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct TwoIndexedModalToNodalMatrixGenerator {
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
    const auto& collocation_pts =
        collocation_points<BasisType, QuadratureType>(num_points);
    // note the integer division
    const size_t spectral_size = (N - m) / 2 + 1;
    Matrix vandermonde_matrix(num_points, spectral_size);
    for (size_t j = m, k = 0; j <= N; ++k, j += 2) {
      const auto& basis_function_values =
          compute_basis_function_value<BasisType>(j, m, collocation_pts);
      for (size_t i = 0; i < num_points; ++i) {
        vandermonde_matrix(i, k) = basis_function_values[i];
      }
    }
    return vandermonde_matrix;
  }
};
}  // namespace

PRECOMPUTED_SPECTRAL_QUANTITY(modal_to_nodal_matrix, Matrix,
                              ModalToNodalMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

SPECTRAL_QUANTITY_FOR_MESH(modal_to_nodal_matrix, Matrix)

#undef SPECTRAL_QUANTITY_FOR_MESH

PRECOMPUTED_TWO_INDEXED_SPECTRAL_QUANTITY(modal_to_nodal_matrix, Matrix,
                                          TwoIndexedModalToNodalMatrixGenerator)

#undef PRECOMPUTED_TWO_INDEXED_SPECTRAL_QUANTITY

TWO_INDEXED_SPECTRAL_QUANTITY_FOR_MESH(modal_to_nodal_matrix, Matrix)

#undef TWO_INDEXED_SPECTRAL_QUANTITY_FOR_MESH

template const Matrix&
    modal_to_nodal_matrix<Basis::Cartoon, Quadrature::AxialSymmetry>(size_t);
template const Matrix& modal_to_nodal_matrix<
    Basis::Cartoon, Quadrature::SphericalSymmetry>(size_t);
template const Matrix&
    modal_to_nodal_matrix<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const Matrix&
    modal_to_nodal_matrix<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const Matrix&
    modal_to_nodal_matrix<Basis::Fourier, Quadrature::Equiangular>(size_t);
template const Matrix&
    modal_to_nodal_matrix<Basis::Legendre, Quadrature::Gauss>(size_t);
template const Matrix&
    modal_to_nodal_matrix<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
template const Matrix& modal_to_nodal_matrix<
    Basis::ZernikeB1, Quadrature::GaussRadauUpper>(size_t, size_t, size_t);
template const Matrix& modal_to_nodal_matrix<
    Basis::ZernikeB2, Quadrature::GaussRadauUpper>(size_t, size_t, size_t);
template const Matrix& modal_to_nodal_matrix<
    Basis::ZernikeB3, Quadrature::GaussRadauUpper>(size_t, size_t, size_t);
// Some compilers require these instantiations to exist
template const Matrix& modal_to_nodal_matrix<Basis::FiniteDifference,
                                             Quadrature::CellCentered>(size_t);
template const Matrix& modal_to_nodal_matrix<Basis::FiniteDifference,
                                             Quadrature::FaceCentered>(size_t);
template const Matrix& modal_to_nodal_matrix<
    Basis::ZernikeB1, Quadrature::GaussRadauUpper>(size_t);
template const Matrix& modal_to_nodal_matrix<
    Basis::ZernikeB2, Quadrature::GaussRadauUpper>(size_t);
template const Matrix& modal_to_nodal_matrix<
    Basis::ZernikeB3, Quadrature::GaussRadauUpper>(size_t);
}  // namespace Spectral
