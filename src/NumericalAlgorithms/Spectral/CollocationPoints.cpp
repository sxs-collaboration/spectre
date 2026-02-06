// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SpectralQuantityForMesh.hpp"

namespace Spectral {

template <Basis BasisType, Quadrature QuadratureType>
const DataVector& collocation_points(const size_t num_points) {
  return detail::precomputed_spectral_quantity<
             BasisType, QuadratureType,
             detail::CollocationPointsAndWeightsGenerator<BasisType,
                                                          QuadratureType>>(
             num_points)
      .first;
}

SPECTRAL_QUANTITY_FOR_MESH(collocation_points, DataVector)

#undef SPECTRAL_QUANTITY_FOR_MESH

template const DataVector&
    collocation_points<Basis::Cartoon, Quadrature::AxialSymmetry>(size_t);
template const DataVector&
    collocation_points<Basis::Cartoon, Quadrature::SphericalSymmetry>(size_t);
template const DataVector&
    collocation_points<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const DataVector&
    collocation_points<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const DataVector&
    collocation_points<Basis::Legendre, Quadrature::Gauss>(size_t);
template const DataVector&
    collocation_points<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
template const DataVector& collocation_points<Basis::FiniteDifference,
                                              Quadrature::CellCentered>(size_t);
template const DataVector& collocation_points<Basis::FiniteDifference,
                                              Quadrature::FaceCentered>(size_t);
template const DataVector&
    collocation_points<Basis::Fourier, Quadrature::Equiangular>(size_t);
template const DataVector&
    collocation_points<Basis::ZernikeB1, Quadrature::GaussRadauUpper>(size_t);
template const DataVector&
    collocation_points<Basis::ZernikeB2, Quadrature::GaussRadauUpper>(size_t);
template const DataVector&
    collocation_points<Basis::ZernikeB3, Quadrature::GaussRadauUpper>(size_t);
}  // namespace Spectral
