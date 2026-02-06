// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BarycentricWeights.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral {
namespace {
template <Basis BasisType, Quadrature QuadratureType>
struct BarycentricWeightsGenerator {
  DataVector operator()(const size_t num_points) const {
    // Algorithm 30 in Kopriva, p. 75
    // This is valid for any collocation points.
    const DataVector& x =
        collocation_points<BasisType, QuadratureType>(num_points);
    DataVector bary_weights(num_points, 1.);
    for (size_t j = 1; j < num_points; j++) {
      for (size_t k = 0; k < j; k++) {
        bary_weights[k] *= x[k] - x[j];
        bary_weights[j] *= x[j] - x[k];
      }
    }
    for (size_t j = 0; j < num_points; j++) {
      bary_weights[j] = 1. / bary_weights[j];
    }
    return bary_weights;
  }
};
}  // namespace

namespace detail {
template <Basis BasisType, Quadrature QuadratureType>
const DataVector& barycentric_weights(const size_t num_points) {
  return precomputed_spectral_quantity<
      BasisType, QuadratureType,
      BarycentricWeightsGenerator<BasisType, QuadratureType>>(num_points);
}

template const DataVector&
    barycentric_weights<Basis::Cartoon, Quadrature::AxialSymmetry>(size_t);
template const DataVector&
    barycentric_weights<Basis::Cartoon, Quadrature::SphericalSymmetry>(size_t);
template const DataVector&
    barycentric_weights<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const DataVector&
    barycentric_weights<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const DataVector&
    barycentric_weights<Basis::Legendre, Quadrature::Gauss>(size_t);
template const DataVector&
    barycentric_weights<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
template const DataVector&
    barycentric_weights<Basis::ZernikeB1, Quadrature::GaussRadauUpper>(size_t);
template const DataVector&
    barycentric_weights<Basis::ZernikeB2, Quadrature::GaussRadauUpper>(size_t);
template const DataVector&
    barycentric_weights<Basis::ZernikeB3, Quadrature::GaussRadauUpper>(size_t);
}  // namespace detail
}  // namespace Spectral
