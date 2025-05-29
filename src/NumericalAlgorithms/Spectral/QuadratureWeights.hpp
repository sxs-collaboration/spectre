// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

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
/*!
 * \brief Weights to compute definite integrals.
 *
 * \details These are the coefficients to contract with the nodal
 * function values \f$f_k\f$ to approximate the definite integral \f$I[f]=\int
 * f(x)\mathrm{d}x\f$.
 *
 * Note that the term _quadrature_ also often refers to the quantity
 * \f$Q[f]=\int f(x)w(x)\mathrm{d}x\approx \sum_k f_k w_k\f$. Here, \f$w(x)\f$
 * denotes the basis-specific weight function w.r.t. to which the basis
 * functions \f$\Phi_k\f$ are orthogonal, i.e \f$\int\Phi_i(x)\Phi_j(x)w(x)=0\f$
 * for \f$i\neq j\f$. The weights \f$w_k\f$ approximate this inner product. To
 * approximate the definite integral \f$I[f]\f$ we must employ the
 * coefficients \f$\frac{w_k}{w(\xi_k)}\f$ instead, where the \f$\xi_k\f$ are
 * the collocation points. These are the coefficients this function returns.
 * Only for a unit weight function \f$w(x)=1\f$, i.e. a Legendre basis, is
 * \f$I[f]=Q[f]\f$ so this function returns the \f$w_k\f$ identically.
 *
 * For a `FiniteDifference` basis or `CellCentered` and `FaceCentered`
 * quadratures, the interpretation of the quadrature weights in term
 * of an approximation to \f$I(q)\f$ remains correct, but its explanation
 * in terms of orthonormal basis is not, i.e. we set \f$w_k\f$ to the grid
 * spacing at each point, and the inverse weight \f$\frac{1}{w(\xi_k)}=1\f$ to
 * recover the midpoint method for definite integrals.
 *
 * \param num_points The number of collocation points
 */
template <Basis BasisType, Quadrature QuadratureType>
const DataVector& quadrature_weights(size_t num_points);

/*!
 * \brief Quadrature weights for a one-dimensional mesh.
 *
 * \see quadrature_weights(size_t)
 */
const DataVector& quadrature_weights(const Mesh<1>& mesh);
}  // namespace Spectral
