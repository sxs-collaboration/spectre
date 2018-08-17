// Distributed under the MIT License.
// See LICENSE.txt for details.

/*!
 * \file
 * Declares functionality to retrieve spectral quantities associated with
 * a particular choice of basis functions and quadrature.
 */

#pragma once

#include <cstddef>
#include <iosfwd>
#include <limits>
#include <utility>

/// \cond
class Matrix;
class DataVector;
template <size_t>
class Mesh;
/// \endcond

/*!
 * \ingroup SpectralGroup
 * \brief Functionality associated with a particular choice of basis functions
 * and quadrature for spectral operations.
 *
 * \details The functions in this namespace provide low-level access to
 * collocation points, quadrature weights and associated matrices, such as for
 * differentiation and interpolation. They are available in two versions: either
 * templated directly on the enum cases of the Spectral::Basis and
 * Spectral::Quadrature types, or taking a one-dimensional Mesh as their
 * argument.
 *
 * \note Generally you should prefer working with a Mesh. Use its
 * Mesh::slice_through() method to retrieve the mesh in a particular dimension:
 * \snippet Test_Spectral.cpp get_points_for_mesh
 *
 *
 * Most algorithms in this namespace are adapted from \cite Kopriva.
 */
namespace Spectral {

/*!
 * \ingroup SpectralGroup
 * \brief The choice of basis functions for computing collocation points and
 * weights.
 *
 * \details Choose `Legendre` for a general-purpose DG mesh, unless you have a
 * particular reason for choosing another basis.
 */
enum class Basis { Chebyshev, Legendre };

/// \cond HIDDEN_SYMBOLS
std::ostream& operator<<(std::ostream& os, const Basis& basis) noexcept;
/// \endcond

/*!
 * \ingroup SpectralGroup
 * \brief The choice of quadrature method to compute integration weights.
 *
 * \details Integrals using \f$N\f$ collocation points with Gauss quadrature are
 * exact to polynomial order \f$p=2N-1\f$. Gauss-Lobatto quadrature is exact
 * only to polynomial order \f$p=2N-3\f$, but includes collocation points at the
 * domain boundary.
 */
enum class Quadrature { Gauss, GaussLobatto };

/// \cond HIDDEN_SYMBOLS
std::ostream& operator<<(std::ostream& os,
                         const Quadrature& quadrature) noexcept;
/// \endcond

/*!
 * \brief Minimum number of possible collocation points for a quadrature type.
 *
 * \details Since Gauss-Lobatto quadrature has points on the domain boundaries
 * it must have at least two collocation points. Gauss quadrature can have only
 * one collocation point.
 */
template <Basis, Quadrature>
constexpr size_t minimum_number_of_points{std::numeric_limits<size_t>::max()};
/// \cond
template <Basis BasisType>
constexpr size_t minimum_number_of_points<BasisType, Quadrature::Gauss> = 1;
template <Basis BasisType>
constexpr size_t minimum_number_of_points<BasisType, Quadrature::GaussLobatto> =
    2;
/// \endcond

/*!
 * \brief Maximum number of allowed collocation points.
 */
template <Basis>
constexpr size_t maximum_number_of_points = 12;

/*!
 * \brief Compute the function values of the basis function \f$\Phi_k(x)\f$
 * (zero-indexed).
 */
template <Basis BasisType>
DataVector compute_basis_function_values(size_t k,
                                         const DataVector& x) noexcept;

/*!
 * \brief Compute the inverse of the weight function \f$w(x)\f$ w.r.t. which
 * the basis functions are orthogonal. See the description of
 * `quadrature_weights(size_t)` for details.
 */
template <Basis>
DataVector compute_inverse_weight_function_values(const DataVector&) noexcept;

/*!
 * \brief Compute the normalization square of the basis function \f$\Phi_k\f$
 * (zero-indexed), i.e. the weighted definite integral over its square.
 */
template <Basis BasisType>
double compute_basis_function_normalization_square(size_t k) noexcept;

/*!
 * \brief Compute the collocation points and weights associated to the
 * basis and quadrature.
 *
 * \details This function is expected to return the tuple
 * \f$(\xi_k,w_k)\f$ where the \f$\xi_k\f$ are the collocation
 * points and the \f$w_k\f$ are defined in the description of
 * `quadrature_weights(size_t)`.
 */
template <Basis BasisType, Quadrature QuadratureType>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights(
    size_t num_points) noexcept;

/*!
 * \brief Collocation points
 * \param num_points The number of collocation points
 */
template <Basis BasisType, Quadrature QuadratureType>
const DataVector& collocation_points(size_t num_points) noexcept;

/*!
 * \brief Collocation points for a one-dimensional mesh.
 *
 * \see collocation_points(size_t)
 */
const DataVector& collocation_points(const Mesh<1>& mesh) noexcept;

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
 * \param num_points The number of collocation points
 */
template <Basis BasisType, Quadrature QuadratureType>
const DataVector& quadrature_weights(size_t num_points) noexcept;

/*!
 * \brief Quadrature weights for a one-dimensional mesh.
 *
 * \see quadrature_weights(size_t)
 */
const DataVector& quadrature_weights(const Mesh<1>& mesh) noexcept;

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
 * \param num_points The number of collocation points
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& differentiation_matrix(size_t num_points) noexcept;

/*!
 * \brief Differentiation matrix for a one-dimensional mesh.
 *
 * \see differentiation_matrix(size_t)
 */
const Matrix& differentiation_matrix(const Mesh<1>& mesh) noexcept;

/*!
 * \brief %Matrix used to interpolate to the \p target_points.
 *
 * \warning It is expected but not checked that the \p target_points are inside
 * the interval covered by the `BasisType` in logical coordinates.
 *
 * \param num_points The number of collocation points
 * \param target_points The points to interpolate to
 */
template <Basis BasisType, Quadrature QuadratureType, typename T>
Matrix interpolation_matrix(size_t num_points, const T& target_points) noexcept;

/*!
 * \brief Interpolation matrix to the \p target_points for a one-dimensional
 * mesh.
 *
 * \see interpolation_matrix(size_t, const T&)
 */
template <typename T>
Matrix interpolation_matrix(const Mesh<1>& mesh,
                            const T& target_points) noexcept;

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

 * \see grid_points_to_spectral_matrix(size_t)
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& spectral_to_grid_points_matrix(size_t num_points) noexcept;

/*!
 * \brief Transformation matrix from modal to nodal coefficients for a
 * one-dimensional mesh.
 *
 * \see spectral_to_grid_points_matrix(size_t)
 */
const Matrix& spectral_to_grid_points_matrix(const Mesh<1>& mesh) noexcept;

/*!
 * \brief %Matrix used to transform from the nodal coefficients of a function to
 * its spectral coefficients (modes). Also referred to as the inverse
 * _Vandermonde matrix_.
 *
 * \details This is the inverse to the Vandermonde matrix \f$\mathcal{V}\f$
 * computed in spectral_to_grid_points_matrix(size_t). It can be computed
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
 * \see spectral_to_grid_points_matrix(size_t)
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& grid_points_to_spectral_matrix(size_t num_points) noexcept;

/*!
 * \brief Transformation matrix from nodal to modal coefficients for a
 * one-dimensional mesh.
 *
 * \see grid_points_to_spectral_matrix(size_t)
 */
const Matrix& grid_points_to_spectral_matrix(const Mesh<1>& mesh) noexcept;

/*!
 * \brief %Matrix used to linearize a function.
 *
 * \details Filters out all except the lowest two modes by applying
 * \f$\mathcal{V}^{-1}\cdot\mathrm{diag}(1,1,0,0,...)\cdot\mathcal{V}\f$ to the
 * nodal coefficients, where \f$\mathcal{V}\f$ is the Vandermonde matrix
 * computed in `spectral_to_grid_points_matrix(size_t)`.
 *
 * \param num_points The number of collocation points
 *
 * \see spectral_to_grid_points_matrix(size_t)
 * \see grid_points_to_spectral_matrix(size_t)
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& linear_filter_matrix(size_t num_points) noexcept;

/*!
 * \brief Linear filter matrix for a one-dimensional mesh.
 *
 * \see linear_filter_matrix(size_t)
 */
const Matrix& linear_filter_matrix(const Mesh<1>& mesh) noexcept;

}  // namespace Spectral
