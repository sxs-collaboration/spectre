// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class Matrix;
template <size_t>
class Mesh;
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
enum class Parity : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {
/// @{
/*!
 * \brief %Matrix used to compute the derivative of a function.
 *
 * \details For a function represented by the nodal coefficients \f$u_j\f$ a
 * matrix multiplication with the differentiation matrix \f$D_{ij}\f$ gives the
 * coefficients of the function's derivative. Since \f$u(x)\f$ is expanded in
 * Lagrange polynomials \f$u(x)=\sum_j u_j l_j(x)\f$ the differentiation matrix
 * is computed as \f$D_{ij}=l_j^\prime(\xi_i)\f$ where the \f$\xi_i\f$ are the
 * collocation points.
 *
 * The finite difference matrix uses summation by parts operators,
 * \f$D_{2-1}, D_{4-2}, D_{4-3}\f$, and \f$D_{6-5}\f$ from \cite Diener2005tn.
 *
 * \param num_points The number of collocation points
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& differentiation_matrix(size_t num_points);
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& differentiation_matrix_transpose(size_t num_points);
/// @}

/*!
 * \brief %Matrix used to compute the derivative of a function of known parity
 *
 * \details For the Zernike basis, defined on \f$[0,1]\f$, `GaussRadauUpper`
 * quadratrue shifts the collocation points to the upper side, which
 * contributes to inaccurate differentiation at the lower side due to the low
 * density of points. By knowing the parity of functions in this basis as it
 * has two indices, we can extend the function to the negative \f$r\f$,
 * greatly reducing errors.
 *
 * \param num_points The number of collocation points
 * \param parity The Parity of the function
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& differentiation_matrix(size_t num_points, Parity parity);

/// @{
/*!
 * \brief Differentiation matrix for a one-dimensional mesh.
 *
 * \see differentiation_matrix(size_t)
 */
const Matrix& differentiation_matrix(const Mesh<1>& mesh);
const Matrix& differentiation_matrix_transpose(const Mesh<1>& mesh);
/// @}

/*!
 * \brief %Matrix used to compute the divergence of the flux in weak form.
 *
 * This is the transpose of the differentiation matrix multiplied by quadrature
 * weights that appear in DG integrals:
 *
 * \begin{equation}
 * \frac{D^T_{ij}} \frac{w_j}{w_i}
 * \end{equation}
 *
 * \param num_points The number of collocation points
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& weak_flux_differentiation_matrix(size_t num_points);

/*!
 * \brief %Matrix used to compute the divergence of the flux in weak form.
 *
 * \see weak_flux_differentiation_matrix(size_t)
 */
const Matrix& weak_flux_differentiation_matrix(const Mesh<1>& mesh);
}  // namespace Spectral
