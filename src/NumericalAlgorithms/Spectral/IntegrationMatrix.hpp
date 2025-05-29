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
const Matrix& integration_matrix(size_t num_points);

/*!
 * \brief Indefinite integration matrix for a one-dimensional mesh.
 *
 * \see integration_matrix(size_t)
 */
const Matrix& integration_matrix(const Mesh<1>& mesh);
}  // namespace Spectral
