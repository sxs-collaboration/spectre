// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/InverseWeightFunctionValues.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SpectralQuantityForMesh.hpp"

namespace Spectral {
namespace {
template <Basis BasisType, Quadrature QuadratureType>
struct QuadratureWeightsGenerator {
  DataVector operator()(const size_t num_points) const {
    const auto& pts_and_weights = detail::precomputed_spectral_quantity<
        BasisType, QuadratureType,
        detail::CollocationPointsAndWeightsGenerator<BasisType,
                                                     QuadratureType>>(
        num_points);
    return pts_and_weights.second *
           compute_inverse_weight_function_values<BasisType>(
               pts_and_weights.first);
  }
};
}  // namespace

PRECOMPUTED_SPECTRAL_QUANTITY(quadrature_weights, DataVector,
                              QuadratureWeightsGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

SPECTRAL_QUANTITY_FOR_MESH(quadrature_weights, DataVector)

#undef SPECTRAL_QUANTITY_FOR_MESH

template const DataVector&
    quadrature_weights<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const DataVector&
    quadrature_weights<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const DataVector&
    quadrature_weights<Basis::Legendre, Quadrature::Gauss>(size_t);
template const DataVector&
    quadrature_weights<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
template const DataVector& quadrature_weights<Basis::FiniteDifference,
                                              Quadrature::CellCentered>(size_t);
template const DataVector& quadrature_weights<Basis::FiniteDifference,
                                              Quadrature::FaceCentered>(size_t);
}  // namespace Spectral
