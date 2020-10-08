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
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
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
 *
 * \warning The `FiniteDifference` "basis" is used to denote that only the
 * collocation points are defined, but that differentiation, integration, and
 * interpolation schemes are to be chosen locally wherever a `FiniteDifference`
 * mesh is being used. The reason is that there isn't a requirement that the
 * basis and collocation point locations are at all related to the
 * differentiation, integration, or interpolation methods - it is merely a
 * convenient choice in a lot of cases. For `FiniteDifference` we need to choose
 * the order of the scheme (and hence the weights, differentiation matrix,
 * integration weights, and interpolant) locally in space and time to handle
 * discontinuous solutions, hence none of those are defined for the
 * `FiniteDifference` basis.
 */
enum class Basis { Chebyshev, Legendre, FiniteDifference };

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
 *
 * \warning `CellCentered` and `FaceCentered` are intended to be used with the
 * `FiniteDifference` basis (though in principle they could be used with any
 * basis), and thus do not implement differentiation matrices, integration
 * weights, and interpolation matrices.
 */
enum class Quadrature { Gauss, GaussLobatto, CellCentered, FaceCentered };

/// \cond HIDDEN_SYMBOLS
std::ostream& operator<<(std::ostream& os,
                         const Quadrature& quadrature) noexcept;
/// \endcond

namespace detail {
constexpr size_t minimum_number_of_points(
    const Basis /*basis*/, const Quadrature quadrature) noexcept {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (quadrature == Quadrature::Gauss) {
    return 1;
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (quadrature == Quadrature::GaussLobatto) {
    return 2;
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (quadrature == Quadrature::CellCentered) {
    return 1;
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (quadrature == Quadrature::FaceCentered) {
    return 2;
  }
  return std::numeric_limits<size_t>::max();
}
}  // namespace detail

/*!
 * \brief Minimum number of possible collocation points for a quadrature type.
 *
 * \details Since Gauss-Lobatto quadrature has points on the domain boundaries
 * it must have at least two collocation points. Gauss quadrature can have only
 * one collocation point.
 *
 * \details For `CellCentered` the minimum number of points is 1, while for
 * `FaceCentered` it is 2.
 */
template <Basis basis, Quadrature quadrature>
constexpr size_t minimum_number_of_points =
    detail::minimum_number_of_points(basis, quadrature);

/*!
 * \brief Maximum number of allowed collocation points.
 *
 * \details We choose a limit of 23 FD grid points because for DG-subcell the
 * number of points in an element is `2 * (number_dg_points - 1)`. Because there
 * is no way of generically retrieving the maximum number of grid points for
 * a non-FD basis, we need to hard-code both values here. If the number of grid
 * points is increased for the non-FD bases, it should also be increased for the
 * FD basis. Note that for good task-based parallelization 23 grid points is
 * already a fairly large number.
 */
template <Basis basis>
constexpr size_t maximum_number_of_points =
    basis == Basis::FiniteDifference ? 23 : 12;

/*!
 * \brief Compute the function values of the basis function \f$\Phi_k(x)\f$
 * (zero-indexed).
 */
template <Basis BasisType, typename T>
T compute_basis_function_value(size_t k, const T& x) noexcept;

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
 *
 * \warning for a `FiniteDifference` basis or `CellCentered` and `FaceCentered`
 * quadratures only the collocation points are set, the weights are `NaN`.
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
 * \warning for a `FiniteDifference` basis or `CellCentered` and `FaceCentered`
 * quadratures the weights are `NaN`.
 *
 * \param num_points The number of collocation points
 */
template <Basis BasisType, Quadrature QuadratureType>
const DataVector& quadrature_weights(size_t num_points) noexcept;

/*!
 * \brief Quadrature weights for a one-dimensional mesh.
 *
 * \warning for a `FiniteDifference` basis or `CellCentered` and `FaceCentered`
 * quadratures the weights are `NaN`.
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
 * \brief %Matrix used to compute the divergence of the flux in weak form.
 *
 * \param num_points The number of collocation points
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& weak_flux_differentiation_matrix(size_t num_points) noexcept;

/*!
 * \brief %Matrix used to compute the divergence of the flux in weak form.
 *
 * \see weak_flux_differentiation_matrix(size_t)
 */
const Matrix& weak_flux_differentiation_matrix(const Mesh<1>& mesh) noexcept;

/*!
 * \brief %Matrix used to perform an indefinite integral of a function over the
 * logical grid. The left boundary condition is such that the integral is 0 at
 * \f$\xi=-1\f$
 *
 * Currently only Legendre and Chebyshev polynomials are implemented, but we
 * provide a derivation for how to compute the indefinite integration matrix for
 * general Jacobi polynomials.
 *
 * #### Legendre Polynomials
 * The Legendre polynomials have the identity:
 *
 * \f{align*}{
 * P_n(x) = \frac{1}{2n+1}\frac{d}{dx}\left(P_{n+1}(x) - P_{n-1}(x)\right)
 * \f}
 *
 * The goal is to evaluate the integral of a function \f$u\f$ expanded in terms
 * of Legendre polynomials as
 *
 * \f{align*}{
 * u(x) = \sum_{i=0}^{N} c_i P_i(x)
 * \f}
 *
 * We similarly expand the indefinite integral of \f$u\f$ as
 *
 * \f{align*}{
 * \left.\int u(y) dy\right\rvert_{x}=&\sum_{i=0}^N \tilde{c}_i P_i(x) \\
 *   =&\left.\int\sum_{i=1}^{N}\frac{c_i}{2i+1}
 *      \left(P_{i+1}(y)-P_{i-1}(y)\right)dy\right\rvert_{x}
 *      + \tilde{c}_0 P_0(x) \\
 *   =&\sum_{i=1}^{N}\left(\frac{c_{i-1}}{2i-1} - \frac{c_{i+1}}{2i+3}\right)
 *     P_i(x) + \tilde{c}_0 P_0(x)
 * \f}
 *
 * Thus we get that for \f$i>0\f$
 *
 * \f{align*}{
 * \tilde{c}_i=\frac{c_{i-1}}{2i-1}-\frac{c_{i+1}}{2i+3}
 * \f}
 *
 * and \f$\tilde{c}_0\f$ is a constant of integration, which we choose such that
 * the integral is 0 at the left boundary of the domain (\f$x=-1\f$). The
 * condition for this is:
 *
 * \f{align*}{
 *   \tilde{c}_0=\sum_{i=1}^{N}(-1)^{i+1}\tilde{c}_i
 * \f}
 *
 * The matrix returned by this function is the product of the tridiagonal matrix
 * for the \f$\tilde{c}_i\f$ and the matrix for the boundary condition.
 *
 * #### Chebyshev Polynomials
 *
 * A similar derivation leads to the relations:
 *
 * \f{align*}{
 *  \tilde{c}_i=&\frac{c_{i-1}-c_{i+1}}{2i},&\mathrm{if}\;i>1 \\
 *  \tilde{c}_1=&c_0 - \frac{c_2}{2},&\mathrm{if}\;i=1 \\
 * \f}
 *
 * We again have:
 *
 * \f{align*}{
 * \tilde{c}_0=\sum_{i=1}^N(-1)^{i+1}\tilde{c}_i
 * \f}
 *
 * These are then used to define the indefinite integration matrix.
 *
 * #### Jacobi Polynomials
 *
 * For general Jacobi polynomials \f$P^{(\alpha,\beta)}_n(x)\f$ given by
 *
 * \f{align*}{
 *  (1-x)^\alpha(1+x)^\beta P^{(\alpha,\beta)}_n(x)=\frac{(-1)^n}{2^n n!}
 *  \frac{d^n}{dx^n}\left[(1-x)^{\alpha+n}(1+x)^{\beta+n}\right]
 * \f}
 *
 * we have that
 *
 * \f{align*}{
 * P^{(\alpha,\beta)}_n(x)=\frac{d}{dx}\left(
 * b^{(\alpha,\beta)}_{n-1,n}P^{(\alpha,\beta)}_{n-1}(x) +
 * b^{(\alpha,\beta)}_{n,n}P^{(\alpha,\beta)}_n(x) +
 * b^{(\alpha,\beta)}_{n+1,n}P^{(\alpha,\beta)}_{n+1}(x)
 * \right)
 * \f}
 *
 * where
 *
 * \f{align*}{
 * b^{(\alpha,\beta)}_{n-1,n}=&-\frac{1}{n+\alpha+\beta}
 *                            a^{(\alpha,\beta)}_{n-1,n} \\
 * b^{(\alpha,\beta)}_{n,n}=&-\frac{2}{\alpha+\beta}
 *                          a^{(\alpha,\beta)}_{n,n} \\
 * b^{(\alpha,\beta)}_{n+1,n}=&\frac{1}{n+1}
 *                            a^{(\alpha,\beta)}_{n+1,n} \\
 * a^{(\alpha,\beta)}_{n-1,n}=&\frac{2(n+\alpha)(n+\beta)}
 *            {(2n+\alpha+\beta+1)(2n+\alpha+\beta)} \\
 * a^{(\alpha,\beta)}_{n,n}=&-\frac{\alpha^2-\beta^2}
 *            {(2n+\alpha+\beta+2)(2n+\alpha+\beta)} \\
 * a^{(\alpha,\beta)}_{n-1,n}=&\frac{2(n+1)(n+\alpha+\beta+1)}
 *            {(2n+\alpha+\beta+2)(2n+\alpha+\beta+1)}
 * \f}
 *
 * Following the same derivation we get that
 *
 * \f{align*}{
 *   \tilde{c}_i=c_{i+1}b^{(\alpha,\beta)}_{i,i+1}
 *              +c_i b^{(\alpha,\beta)}_{i,i}
 *              +c_{i-1}b^{(\alpha,\beta)}_{i,i-1}
 * \f}
 *
 * and the boundary condition is
 *
 * \f{align*}{
 *  \tilde{c}_0=\sum_{i=1}^N(-1)^{i+1}
 *              \frac{\Gamma(i+\alpha+1)}{i!\Gamma(\alpha+1)} \tilde{c}_i
 * \f}
 *
 * where \f$\Gamma(x)\f$ is the Gamma function.
 */
template <Basis BasisType, Quadrature QuadratureType>
const Matrix& integration_matrix(size_t num_points) noexcept;

/*!
 * \brief Indefinite integration matrix for a one-dimensional mesh.
 *
 * \see integration_matrix(size_t)
 */
const Matrix& integration_matrix(const Mesh<1>& mesh) noexcept;

/*!
 * \brief %Matrix used to interpolate to the \p target_points.
 *
 * \warning For each target point located outside of the logical coordinate
 * interval covered by `BasisType` (often \f$[-1,1]\f$), the resulting matrix
 * performs polynomial extrapolation instead of interpolation. The extapolation
 * will be correct but may suffer from reduced accuracy, especially for
 * higher-order polynomials (i.e., larger values of `num_points`).
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

// @{
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
    const Mesh<1>& mesh) noexcept;

template <Basis BasisType, Quadrature QuadratureType>
const std::pair<Matrix, Matrix>& boundary_interpolation_matrices(
    size_t num_points) noexcept;
// @}

// @{
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
    const Mesh<1>& mesh) noexcept;

template <Basis BasisType, Quadrature QuadratureType>
const std::pair<DataVector, DataVector>& boundary_lifting_term(
    size_t num_points) noexcept;
// @}

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
const Matrix& modal_to_nodal_matrix(size_t num_points) noexcept;

/*!
 * \brief Transformation matrix from modal to nodal coefficients for a
 * one-dimensional mesh.
 *
 * \see modal_to_nodal_matrix(size_t)
 */
const Matrix& modal_to_nodal_matrix(const Mesh<1>& mesh) noexcept;

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
const Matrix& nodal_to_modal_matrix(size_t num_points) noexcept;

/*!
 * \brief Transformation matrix from nodal to modal coefficients for a
 * one-dimensional mesh.
 *
 * \see nodal_to_modal_matrix(size_t)
 */
const Matrix& nodal_to_modal_matrix(const Mesh<1>& mesh) noexcept;

/*!
 * \brief %Matrix used to linearize a function.
 *
 * \details Filters out all except the lowest two modes by applying
 * \f$\mathcal{V}^{-1}\cdot\mathrm{diag}(1,1,0,0,...)\cdot\mathcal{V}\f$ to the
 * nodal coefficients, where \f$\mathcal{V}\f$ is the Vandermonde matrix
 * computed in `modal_to_nodal_matrix(size_t)`.
 *
 * \param num_points The number of collocation points
 *
 * \see modal_to_nodal_matrix(size_t)
 * \see nodal_to_modal_matrix(size_t)
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

/// \cond
template <>
struct Options::create_from_yaml<Spectral::Quadrature> {
  template <typename Metavariables>
  static Spectral::Quadrature create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
Spectral::Quadrature
Options::create_from_yaml<Spectral::Quadrature>::create<void>(
    const Options::Option& options);
/// \endcond
