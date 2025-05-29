// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BoundaryInterpolationTerm.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral {
template <Basis BasisType, Quadrature QuadratureType>
const std::pair<DataVector, DataVector>& boundary_interpolation_term(
    const size_t num_points) {
  static_assert(BasisType == Spectral::Basis::Legendre);
  static_assert(
      QuadratureType == Spectral::Quadrature::Gauss,
      "We only compute the time derivative correction interpolation for Gauss "
      "quadrature since for Gauss-Lobatto you can just copy values off the "
      "volume.");
  static const auto cache = make_static_cache<
      CacheRange<Spectral::minimum_number_of_points<BasisType, QuadratureType>,
                 Spectral::maximum_number_of_points<BasisType> + 1>>(
      [](const size_t local_num_points) {
        const Matrix interp_matrix =
            interpolation_matrix<BasisType, Quadrature::GaussLobatto>(
                local_num_points == 1 ? 2 : local_num_points,
                collocation_points<BasisType, Quadrature::Gauss>(
                    local_num_points));

        std::pair<DataVector, DataVector> result{DataVector{local_num_points},
                                                 DataVector{local_num_points}};
        for (size_t i = 0; i < local_num_points; ++i) {
          result.first[i] = interp_matrix(i, 0);
          result.second[i] = interp_matrix(i, interp_matrix.columns() - 1);
        }
        return result;
      });
  return cache(num_points);
}

const std::pair<DataVector, DataVector>& boundary_interpolation_term(
    const Mesh<1>& mesh) {
  ASSERT(mesh.basis(0) == Spectral::Basis::Legendre,
         "We only support DG with a Legendre basis.");
  ASSERT(
      mesh.quadrature(0) == Spectral::Quadrature::Gauss,
      "We only compute the time derivative correction interpolation for Gauss "
      "quadrature since for Gauss-Lobatto you can just copy values off the "
      "volume.");
  return boundary_interpolation_term<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::Gauss>(
      mesh.extents(0));
}

template const std::pair<DataVector, DataVector>& boundary_interpolation_term<
    Basis::Legendre, Quadrature::Gauss>(const size_t num_points);
}  // namespace Spectral
