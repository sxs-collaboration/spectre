// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/InverseWeightFunctionValues.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/PrecomputedSpectralQuantity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SpectralQuantityForMesh.hpp"

namespace Spectral {
template <Basis BasisType>
Matrix spectral_definite_integral_matrix(size_t num_points);

namespace {
template <Basis BasisType, Quadrature QuadratureType>
struct QuadratureWeightsGenerator {
  DataVector operator()(const size_t num_points) const {
    if constexpr (BasisType == Basis::Chebyshev) {
      DataVector result{num_points};
      const Matrix q =
          spectral_definite_integral_matrix<BasisType>(num_points) *
          Spectral::nodal_to_modal_matrix<BasisType, QuadratureType>(
              num_points);
      for (size_t i = 0; i < num_points; ++i) {
        result[i] = q(0, i);
      }
      return result;
    } else {
      return detail::precomputed_spectral_quantity<
                 BasisType, QuadratureType,
                 detail::CollocationPointsAndWeightsGenerator<BasisType,
                                                              QuadratureType>>(
                 num_points)
          .second;
    }
  }
};
}  // namespace

PRECOMPUTED_SPECTRAL_QUANTITY(quadrature_weights, DataVector,
                              QuadratureWeightsGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

SPECTRAL_QUANTITY_FOR_MESH(quadrature_weights, DataVector)

#undef SPECTRAL_QUANTITY_FOR_MESH

template const DataVector&
quadrature_weights<Basis::Cartoon, Quadrature::AxialSymmetry>(size_t);
template const DataVector&
quadrature_weights<Basis::Cartoon, Quadrature::SphericalSymmetry>(size_t);
template const DataVector&
quadrature_weights<Basis::Chebyshev, Quadrature::Gauss>(size_t);
template const DataVector&
quadrature_weights<Basis::Chebyshev, Quadrature::GaussLobatto>(size_t);
template const DataVector&
quadrature_weights<Basis::Legendre, Quadrature::Gauss>(size_t);
template const DataVector&
quadrature_weights<Basis::Legendre, Quadrature::GaussLobatto>(size_t);
template const DataVector&
quadrature_weights<Basis::FiniteDifference, Quadrature::CellCentered>(size_t);
template const DataVector&
quadrature_weights<Basis::FiniteDifference, Quadrature::FaceCentered>(size_t);
template const DataVector&
quadrature_weights<Basis::Fourier, Quadrature::Equiangular>(size_t);
}  // namespace Spectral
