// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
class Matrix;
namespace Spectral {
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {

/*!
 * \ingroup SpectralGroup
 *
 * \brief A collection of helper functions for Chebyshev polynomials
 *
 * \details The Chebyshev polynomials are given by:
 * \f[
 * T_n (x) = \cos(n \theta), \text{      } \theta = \arccos(x)
 * \f]
 *
 * The Chebyshev expansion of a function \f$f \in [-1,1] \f$ is given by:
 * \f[
 * f(x) = \sum_{n=0}^{\infty} f_n T_n(x)
 * \f]
 * where
 * \f[
 * f_n = \frac{1}{c_n} \int_{-1}^1 f(x) T_n(x) w(x) dx
 * \f]
 * where the weight function is \f$w(x) = (1-x^2)^{-1/2}\f$ and the basis
 * function normalization value is given by:
 * \f{align*}{
 *   c_n
 *   &=\begin{cases}
 *     \hfil \pi \hfil & \text{if } n = 0 \\
 *     \hfil \frac{\pi}{2} \hfil & \text{otherwise}
 *   \end{cases}
 * \f}
 *
 * If a function is discretized at \f$N+1\f$ collocation points, the modal
 * representation will have \f$N+1\f$ spectral coefficients consisting of
 * \f[
 * f_n \qquad \text{for } n = 0, \ldots, N
 * \f]
 *
 * For more details about using Chebyshev polynomials see e.g. \cite Boyd2001
 * and \cite Fornberg1996.
 */
class Chebyshev {
 public:
  /*!
   * \brief Value of the basis function \f$\Phi_k(x) = T_k(x)\f$
   */
  template <typename T>
  static T basis_function_value(size_t k, const T& x);

  /*!
   * \brief The normalization square \f$c_k\f$ of the basis function
   * \f$\Phi_k(x)\f$, i.e. the definite integral of its square.
   */
  static double basis_function_normalization_square(size_t k);

  /*!
   * \brief Collocation points \f$\{x_i\}\f$
   *
   * \details The collocation points on the interval \f$[-1, 1]\f$ are given by
   * \f{align*}{
   *   x_i
   *   &=\begin{cases}
   *     \hfil - \cos \frac{(2i+1)\pi}{2N+2} \hfil &
   *       \text{for Quadrature::Gauss} \\
   *     \hfil - \cos \frac{i \pi}{N} \hfil &
   *       \text{for Quadrature::GaussLobatto}
   *   \end{cases}
   * \f}
   */
  template <Quadrature quadrature>
  static DataVector collocation_points(size_t num_points);

  /*!
   * \brief Integration weights \f$\{w_i\}\f$
   *
   * \details The integration weights are used to approximate the weighted
   * integral \f$Q[f]=\int f(x)w(x)\mathrm{d}x\approx \sum_k f_k w_k\f$.
   * For Quadrature::Gauss, the weights are given by:
   * \f[
   * w_i = \frac{\pi}{N+1}
   * \f]
   *
   * For Quadrature::GaussLobatto, the weights are given by:
   * \f{align*}{
   *   w_i
   *   &=\begin{cases}
   *     \hfil \frac{\pi}{2N}  \hfil & \text{for } j = 0, N\\
   *     \hfil \frac{\pi}{N}   \hfil & \text{for } j = 1, \ldots, N-1
   *   \end{cases}
   * \f}
   *
   * \note These weights are used to compute the modes from the nodal
   * values.  They are not used to evaluate definite or indefinite
   * integrals directly from the nodal values.
   */
  template <Quadrature quadrature>
  static DataVector integration_weights(size_t num_points);

  /*!
   * \brief Matrix used to compute the modes of the indefinite
   * integral from the modes of the integrand such that the constant
   * of integration is determined by requiring the integral to be zero
   * at \f$x=-1\f$.
   *
   * \details Chebyshev polynomials satisfy the identity:
   * \f[
   * \int^x dy T_n (y) = \left\{ \frac{T_{n+1}(x)}{2(n+1)} +
   *   \frac{T_{n-1}(x)}{2(n-1)} \right\}, \qquad n \geq 2
   * \f]
   *
   * Thus the modes \f$\tilde{f}_j\f$ of the integral are given as:
   * \f{align*}{
   *   \tilde{f}_i
   *   &=\begin{cases}
   *     \hfil \frac{f_{i-1}-f_{i+1}}{2i}, \hfil & \text{for } i > 1\\
   *     \hfil f_0 - \frac{f_2}{2} \hfil & \text{for } i = 1
   *   \end{cases}
   * \f}
   * where \f$f_j\f$ are the modes of the integrand.
   *\f$\tilde{f}_0\f$ is a constant of integration, which we choose such that
   * the integral is 0 at the left boundary of the domain (\f$x=-1\f$). The
   * condition for this is:
   *
   * \f[
   *   \tilde{f}_0=\sum_{i=1}^{N}(-1)^{i+1}\tilde{f}_i
   * \f]
   */
  static Matrix indefinite_integral_matrix(size_t num_points);

  /*!
   * \brief Row-vector used to compute the definite integral from the modes of
   * the integrand
   *
   * \details Given the modes \f$\tilde{f}_j\f$ of the integral (see
   * indefinite_integral_matrix), the definite integral is given by evaluating
   * the series at \f$x=1\f$ and \f$x=-1\f$ and taking the difference.  Given
   * that \f$T_n(1) = 1\f$ and \f$T_n(-1) = (-1)^n\f$, this means multiplying
   * the indefinite_integral_matrix by the row-vector \f$\{0, 2, 0, 2, \ldots
   * \}\f$ which yields:
   * \f{align*}{
   *   \tilde{q}_i
   *   &=\begin{cases}
   *     \hfil -\frac{2}{n^2-1}, \hfil & \text{for } i \text{ even}\\
   *     \hfil 0 \hfil & \text{for } i \text{ odd}
   *   \end{cases}
   * \f}
   */
  static Matrix definite_integral_matrix(size_t num_points);
};
}  // namespace Spectral
