// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

/// \cond
class Matrix;
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
 * \brief Matrices that interpolate to the lower and upper boundaries of the
 * element.
 *
 * Assumes that the logical coordinates are \f$[-1, 1]\f$. The first element of
 * the pair interpolates to \f$\xi=-1\f$ and the second to \f$\xi=1\f$. These
 * are just the Lagrange interpolating polynomials evaluated at \f$\xi=\pm1\f$.
 * For Gauss-Lobatto points the only non-zero element is at the boundaries
 * and is one and so is not implemented.
 *
 * \warning This can only be called with Gauss points.
 */
const std::pair<Matrix, Matrix>& boundary_interpolation_matrices(
    const Mesh<1>& mesh);

template <Basis BasisType, Quadrature QuadratureType>
const std::pair<Matrix, Matrix>& boundary_interpolation_matrices(
    size_t num_points);
/// @}
}  // namespace Spectral
