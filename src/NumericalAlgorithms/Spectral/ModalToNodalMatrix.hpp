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
}  // namespace Spectral
/// \endcond

namespace Spectral {
/*!
 * \brief %Matrix used to transform from the spectral coefficients (modes) of a
 * function to its nodal coefficients. Also referred to as the _Vandermonde
 * matrix_.
 *
 * \details The Vandermonde matrix is computed as
 * \f$\mathcal{V}_{ij}=\Phi_j(\xi_i)\f$ where the \f$\Phi_j(x)\f$ are the
 * spectral basis functions used in the modal expansion
 * \f$u(x)=\sum_j \widetilde{u}_j\Phi_j(x)\f$, e.g. normalized Legendre
 * polynomials, and the \f$\xi_i\f$ are the collocation points used to construct
 * the interpolating Lagrange polynomials in the nodal expansion
 * \f$u(x)=\sum_j u_j l_j(x)\f$. Then the Vandermonde matrix arises as
 * \f$u(\xi_i)=u_i=\sum_j \widetilde{u}_j\Phi_j(\xi_i)=\sum_j
 * \mathcal{V}_{ij}\widetilde{u}_j\f$.
 *
 * \param num_points The number of collocation points

 * \see nodal_to_modal_matrix(size_t)
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& modal_to_nodal_matrix(size_t num_points);

/*!
 * \brief %Matrix used to transform from the spectral coefficients (modes) of a
 * function to its nodal coefficients. This two-index version is used for
 * two-dimensional basis function (i.e. Zernike with GaussRadauUpper
 * quadrature).
 *
 * For Zernike, \f$m\f$ is the angular index and \f$N\f$ is the
 * maximum supported spectral index (usually taken to be the maximum value,
 * \f$2 \, \texttt{num_points}-2\f$). Note that the size of a spectral space
 * vector is \f$(N - m) / 2 + 1\f$, using integer division.
 *
 * \see modal_to_nodal_matrix(size_t)
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& modal_to_nodal_matrix(size_t num_points, size_t m, size_t N);

/*!
 * \brief Transformation matrix from modal to nodal coefficients for a
 * one-dimensional mesh.
 *
 * \see modal_to_nodal_matrix(size_t)
 */
const Matrix& modal_to_nodal_matrix(const Mesh<1>& mesh);

/*!
 * \brief Transformation matrix from modal to nodal coefficients for a
 * one-dimensional mesh with a Zernike basis.
 *
 * \see modal_to_nodal_matrix(size_t, size_t, size_t)
 */
const Matrix& modal_to_nodal_matrix(const Mesh<1>& mesh, size_t m, size_t N);
}  // namespace Spectral
