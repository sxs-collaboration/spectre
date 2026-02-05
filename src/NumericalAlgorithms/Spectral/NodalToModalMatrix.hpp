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
 * \brief %Matrix used to transform from the nodal coefficients of a function to
 * its spectral coefficients (modes). Also referred to as the inverse
 * _Vandermonde matrix_.
 *
 * \details This is the inverse to the Vandermonde matrix \f$\mathcal{V}\f$
 * computed in modal_to_nodal_matrix(size_t). It can be computed
 * analytically for Gauss quadrature by evaluating
 * \f$\sum_j\mathcal{V}^{-1}_{ij}u_j=\widetilde{u}_i=
 * \frac{(u,\Phi_i)}{\gamma_i}\f$
 * for a Lagrange basis function \f$u(x)=l_k(x)\f$ to find
 * \f$\mathcal{V}^{-1}_{ij}=\mathcal{V}_{ji}\frac{w_j}{\gamma_i}\f$ where the
 * \f$w_j\f$ are the Gauss quadrature weights and \f$\gamma_i\f$ is the norm
 * square of the spectral basis function \f$\Phi_i\f$.
 *
 * \param num_points The number of collocation points
 *
 * \see modal_to_nodal_matrix(size_t)
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& nodal_to_modal_matrix(size_t num_points);

/*!
 * \brief %Matrix used to transform from the nodal coefficients of a function to
 * its spectral coefficients (modes). This two-index version is used for
 * two-dimensional basis function (i.e. Zernike with GaussRadauUpper
 * quadrature).
 *
 * For Zernike, \f$m\f$ is the angular index and \f$N\f$ is the
 * maximum supported spectral index (usually taken to be the maximum value,
 * \f$2 \, \texttt{num_points}-2\f$). Note that the size of a spectral space
 * vector is \f$(N - m) / 2 + 1\f$, using integer division.
 *
 * \see nodal_to_modal_matrix(size_t)
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& nodal_to_modal_matrix(size_t num_points, size_t m, size_t N);

/*!
 * \brief Transformation matrix from nodal to modal coefficients for a
 * one-dimensional mesh.
 *
 * \see nodal_to_modal_matrix(size_t)
 */
const Matrix& nodal_to_modal_matrix(const Mesh<1>& mesh);

/*!
 * \brief Transformation matrix from nodal to modal coefficients for a
 * one-dimensional mesh with a Zernike basis.
 *
 * \see nodal_to_modal_matrix(size_t, size_t, size_t)
 */
const Matrix& nodal_to_modal_matrix(const Mesh<1>& mesh, size_t m, size_t N);
}  // namespace Spectral
