// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

/// \cond
class DataVector;
template <size_t>
class Mesh;
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {
/// @{
/*!
 * \brief Terms used during the lifting portion of a discontinuous Galerkin
 * scheme when using Gauss points.
 *
 * Assumes that the logical coordinates are \f$[-1, 1]\f$. The first element of
 * the pair is the Lagrange polyonmials evaluated at \f$\xi=-1\f$ divided by the
 * weights and the second at \f$\xi=1\f$. Specifically,
 *
 * \f{align*}{
 * \frac{\ell_j(\xi=\pm1)}{w_j}
 * \f}
 *
 * \warning This can only be called with Gauss points.
 */
const std::pair<DataVector, DataVector>& boundary_lifting_term(
    const Mesh<1>& mesh);

template <Basis BasisType, Quadrature QuadratureType>
const std::pair<DataVector, DataVector>& boundary_lifting_term(
    size_t num_points);
/// @}
}  // namespace Spectral
