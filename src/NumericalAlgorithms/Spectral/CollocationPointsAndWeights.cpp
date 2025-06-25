// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral::detail {
template <Basis BasisType, Quadrature QuadratureType>
std::pair<DataVector, DataVector>
CollocationPointsAndWeightsGenerator<BasisType, QuadratureType>::operator()(
    const size_t num_points) const {
  return compute_collocation_points_and_weights<BasisType, QuadratureType>(
      num_points);
}

template struct CollocationPointsAndWeightsGenerator<Basis::Cartoon,
                                                     Quadrature::AxialSymmetry>;
template struct CollocationPointsAndWeightsGenerator<
    Basis::Cartoon, Quadrature::SphericalSymmetry>;
template struct CollocationPointsAndWeightsGenerator<Basis::Chebyshev,
                                                     Quadrature::Gauss>;
template struct CollocationPointsAndWeightsGenerator<Basis::Chebyshev,
                                                     Quadrature::GaussLobatto>;
template struct CollocationPointsAndWeightsGenerator<Basis::Legendre,
                                                     Quadrature::Gauss>;
template struct CollocationPointsAndWeightsGenerator<Basis::Legendre,
                                                     Quadrature::GaussLobatto>;
template struct CollocationPointsAndWeightsGenerator<Basis::FiniteDifference,
                                                     Quadrature::CellCentered>;
template struct CollocationPointsAndWeightsGenerator<Basis::FiniteDifference,
                                                     Quadrature::FaceCentered>;
}  // namespace Spectral::detail
