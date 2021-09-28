// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t Dim>
class Index;
class Matrix;
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::fd {
/*!
 * \ingroup DgSubcellGroup
 * \brief Computes the projection matrix in `Dim` dimensions going from a DG
 * mesh to a conservative finite difference subcell mesh.
 */
template <size_t Dim>
const Matrix& projection_matrix(const Mesh<Dim>& dg_mesh,
                                const Index<Dim>& subcell_extents);

/*!
 * \ingroup DgSubcellGroup
 * \brief Computes the matrix needed for reconstructing the DG solution from
 * the subcell solution.
 *
 * Reconstructing the DG solution from the FD solution is a bit more
 * involved than projecting the DG solution to the FD subcells. Denoting the
 * projection operator by \f$\mathcal{P}\f$ and the reconstruction operator by
 * \f$\mathcal{R}\f$, we desire the property
 *
 * \f{align*}{
 *   \mathcal{R}(\mathcal{P}(u_{\breve{\imath}}
 *   J_{\breve{\imath}}))=u_{\breve{\imath}} J_{\breve{\imath}},
 * \f}
 *
 * where \f$\breve{\imath}\f$ denotes a grid point on the DG grid, \f$u\f$ is
 * the solution on the DG grid, and \f$J\f$ is the determinant of the Jacobian
 * on the DG grid. We also require that the integral of the conserved variables
 * over the subcells is equal to the integral over the DG element. That is,
 *
 * \f{align*}{
 *   \int_{\Omega}u \,d^3x =\int_{\Omega} \underline{u} \,d^3x \Longrightarrow
 *   \int_{\Omega}u J \,d^3\xi=\int_{\Omega} \underline{u} J \,d^3\xi,
 * \f}
 *
 * where \f$\underline{u}\f$ is the solution on the subcells. Because the number
 * of subcell points is larger than the number of DG points, we need to solve a
 * constrained linear least squares problem to reconstruct the DG solution from
 * the subcells.
 *
 * The final reconstruction matrix is given by
 *
 * \f{align*}{
 *   R_{\breve{\jmath}\underline{i}}
 *   &=\left\{(2 \mathcal{P}\otimes\mathcal{P})^{-1}2\mathcal{P} - (2
 *   \mathcal{P}\otimes\mathcal{P})^{-1}\vec{w}\left[\mathbf{w}(2
 *   \mathcal{P}\otimes\mathcal{P})^{-1}\vec{w}\right]^{-1}\mathbf{w}(2
 *   \mathcal{P}\otimes\mathcal{P})^{-1}2\mathcal{P}
 *   + (2 \mathcal{P}\otimes\mathcal{P})^{-1}\vec{w}\left[\mathbf{w}(2
 *   \mathcal{P}\otimes\mathcal{P})^{-1}\vec{w}\right]^{-1}\vec{\underline{w}}
 *   \right\}_{\breve{\jmath}\underline{i}},
 * \f}
 *
 * where \f$\vec{w}\f$ is the vector of integration weights on the DG element,
 * \f$\mathbf{w}=w_{\breve{l}}\delta_{\breve{l}\breve{\jmath}}\f$, and
 * \f$\vec{\underline{w}}\f$ is the vector of integration weights over the
 * subcells. The integration weights \f$\vec{\underline{w}}\f$ on the subcells
 * are those for 6th-order integration on a uniform mesh.
 */
template <size_t Dim>
const Matrix& reconstruction_matrix(const Mesh<Dim>& dg_mesh,
                                    const Index<Dim>& subcell_extents);
}  // namespace evolution::dg::subcellfd
