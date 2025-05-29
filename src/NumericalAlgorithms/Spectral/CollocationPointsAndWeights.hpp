// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

/// \cond
class DataVector;
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {
namespace detail {
template <Basis BasisType, Quadrature QuadratureType>
struct CollocationPointsAndWeightsGenerator {
  std::pair<DataVector, DataVector> operator()(size_t num_points) const;
};
}  // namespace detail

/*!
 * \brief Compute the collocation points and weights associated to the
 * basis and quadrature.
 *
 * \details This function is expected to return the tuple
 * \f$(\xi_k,w_k)\f$ where the \f$\xi_k\f$ are the collocation
 * points and the \f$w_k\f$ are defined in the description of
 * `quadrature_weights(size_t)`.
 *
 * \warning for a `FiniteDifference` basis or `CellCentered` and `FaceCentered`
 * quadratures, the weights are defined to integrate with the midpoint method
 */
template <Basis BasisType, Quadrature QuadratureType>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights(
    size_t num_points);
}  // namespace Spectral
