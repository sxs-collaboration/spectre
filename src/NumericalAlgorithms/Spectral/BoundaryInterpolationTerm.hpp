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
 * \brief Interpolates values from the boundary into the volume, which is needed
 * when applying time derivative or Bjorhus-type boundary conditions in a
 * discontinuous Galerkin scheme using Gauss points.
 *
 * Assumes that the logical coordinates are \f$[-1, 1]\f$.
 * The interpolation is done by assuming the time derivative correction is zero
 * on interior nodes. With a nodal Lagrange polynomial basis this means that
 * only the \f$\ell^{\mathrm{Gauss-Lobatto}}_{0}\f$ and
 * \f$\ell^{\mathrm{Gauss-Lobatto}}_{N}\f$ polynomials/basis functions
 * contribute to the correction. In order to interpolate the correction from the
 * nodes at the boundary, the Gauss-Lobatto Lagrange polynomials  must be
 * evaluated at the Gauss grid points. The returned pair of `DataVector`s stores
 *
 * \f{align*}{
 *   &\ell^{\mathrm{Gauss-Lobatto}}_{0}(\xi_j^{\mathrm{Gauss}}), \\
 *   &\ell^{\mathrm{Gauss-Lobatto}}_{N}(\xi_j^{\mathrm{Gauss}}).
 * \f}
 *
 * This is a different correction from lifting. Lifting is done using the mass
 * matrix, which is an integral over the basis functions, while here we use
 * interpolation.
 *
 * \warning This can only be called with Gauss points.
 */
const std::pair<DataVector, DataVector>& boundary_interpolation_term(
    const Mesh<1>& mesh);

template <Basis BasisType, Quadrature QuadratureType>
const std::pair<DataVector, DataVector>& boundary_interpolation_term(
    size_t num_points);
/// @}
}  // namespace Spectral
