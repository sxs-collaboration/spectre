// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
class Matrix;
/// \endcond

namespace Spectral {

/*!
 * \ingroup SpectralGroup
 *
 * \brief A collection of helper functions for using a Fourier series
 *
 * \details The general real-valued Fourier series is given by:
 * \f[
 * f(x) = a_0 + \sum_{n=1}^{\infty} a_n \cos(nx) + b_n \sin(nx)
 * \f]
 * where
 * \f{align*}
 * a_0 &= \frac{1}{2\pi} \int_0^{2\pi} f(x) dx \\
 * a_n &= \frac{1}{\pi} \int_0^{2\pi} f(x) \cos(nx) dx \\
 * b_n &= \frac{1}{\pi} \int_0^{2\pi} f(x) \sin(nx) dx
 * \f}
 *
 * If a function is discretized at \f$N\f$ collocation points, the modal
 * representation will have \f$N\f$ spectral coefficients consisting of
 * \f{align*}
 * a_n & \qquad \text{for } n = 0, \ldots, \frac{N}{2} \\
 * b_n & \qquad \text{for } n = 1, \ldots, \frac{N-1}{2}
 * \f}
 * Thus to represent all terms through the harmonic \f$M\f$ requires \f$N = 2M
 * +1\f$ collocation points.
 *
 * For more details about using Fourier series see e.g. \cite Boyd2001 and
 * \cite Fornberg1996.
 */
class Fourier {
 public:
  /*!
   * \brief Value of the basis function \f$\Phi_k(x)\f$
   *
   * \details We define the basis functions as
   * \f{align*}{
   *   \Phi_k(x)
   *   &=\begin{cases}
   *     \hfil \cos(kx) \hfil & \text{if } k \geq 0 \\
   *     \hfil \sin(-kx) \hfil & \text{if } k < 0
   *   \end{cases}
   * \f}
   */
  template <typename T>
  static T basis_function_value(int k, const T& x);

  /*!
   * \brief The normalization square \f$c_k\f$ of the basis function
   * \f$\Phi_k(x)\f$, i.e. the definite integral of its square.
   *
   * \details
   * In particular,
   * \f{align*}{
   *   c_k
   *   &=\begin{cases}
   *     \hfil 2 \pi \hfil & \text{if } k = 0 \\
   *     \hfil \pi \hfil & \text{otherwise}
   *   \end{cases}
   * \f}
   */
  static double basis_function_normalization_square(int k);

  /*!
   * \brief Collocation points \f$\{x_i\}\f$
   *
   * \details The collocation points on the interval \f$0 \leq x < 2 \pi\f$ are
   * given by
   * \f[
   * x_i = \frac{2 \pi i}{N}
   * \f]
   */
  static DataVector collocation_points(size_t num_points);

  /*!
   * \brief Quadrature weights \f$\{w_i\}\f$
   *
   * \details The quadrature weights are given by
   * \f[
   * w_i = \frac{2 \pi}{N}
   * \f]
   */
  static DataVector quadrature_weights(size_t num_points);

  /*!
   * \brief Storage index (offset) \f$i\f$ into a ModalVector (representing the
   * coefficients) of the given mode \f$k\f$
   *
   * \details The modal coefficients are stored in a ModalVector as
   * \f$\{u_0, u_1, u_{-1}, u_2, u_{-2}, \ldots, u_M, u_{-M}\}\f$,
   * where \f$u_{-M}\f$ is omitted if the number of coefficients is even.
   * Therefore the storage index \f$i\f$ for the mode \f$u_k\f$ is:
   * \f{align*}{
   *   i
   *   &=\begin{cases}
   *     \hfil 0 \hfil & \text{if } k = 0 \\
   *     \hfil 2k-1 \hfil & \text{if } k > 0 \\
   *     \hfil -2k \hfil & \text{if } k < 0
   *   \end{cases}
   * \f}
   */
  static size_t modal_storage_index(int k);

  /*!
   * \brief Mode \f$k\f$ corresponding to given storage index (offset) \f$i\f$
   * into a ModalVector (representing the coefficients)
   *
   * \details The modal coefficients are stored in a ModalVector as
   * \f$\{u_0, u_1, u_{-1}, u_2, u_{-2}, \ldots, u_M, u_{-M}\}\f$,
   * where \f$u_{-M}\f$ is omitted if the number of coefficients is even.
   * Therefore the storage index \f$i\f$ for the mode \f$u_k\f$ is:
   * \f{align*}{
   *   k
   *   &=\begin{cases}
   *     \hfil 0 \hfil & \text{if } i = 0 \\
   *     \hfil 1 + \frac{i}{2} \hfil & \text{if } i \text{ is odd} \\
   *     \hfil -\frac{i}{2} \hfil & \text{if } i \text{ is even}
   *   \end{cases}
   * \f}
   */
  static int mode_at_storage_index(size_t storage_index);

  /*!
   * \brief Matrix \f$D_{i,j}\f$ used to obtain the first derivative
   *
   * \details The differentiation matrix is given by:
   * \f{align*}{
   *   D_{i,j}
   *   &=\begin{cases}
   *     \hfil 0 \hfil & \text{if } i = j \\
   *     \hfil \frac{1}{2} (-1)^{i-j} \csc \left[0.5 (x_i - x_j)\right] \hfil &
   *     \text{if } i \neq j,\; N \text{ is odd} \\
   *     \hfil \frac{1}{2} (-1)^{i-j} \cot \left[0.5 (x_i - x_j)\right] \hfil &
   *     \text{if } i \neq j,\; N \text{ is even}
   *   \end{cases}
   * \f}
   */
  static Matrix differentiation_matrix(size_t num_points);

  /*!
   * \brief %Matrix used to interpolate to the \p target_points.
   *
   * \details Each row of the matrix is given by the interpolation weights for
   * interpolating to a particular target point \f$x\f$.  At each target point,
   * \f$x\f$, the interpolation weights are given by:
   * \f{align*}{
   *   C_j(x)
   *   &=\begin{cases}
   *     \hfil \frac{1}{N} \sin \left[0.5 N(x - x_j)\right]
   *     \csc \left[0.5 (x - x_j)\right] \hfil & \text{if } N \text{ is odd} \\
   *     \hfil \frac{1}{N} \sin \left[0.5 N (x - x_j)\right]
   *     \cot \left[0.5 (x - x_j)\right] \hfil & \text{if } N \text{ is even}
   *   \end{cases}
   * \f}
   * where \f$x_j\f$ are the collocation_points.
   */
  template <typename T>
  static Matrix interpolation_matrix(size_t num_points, const T& target_points);
};
}  // namespace Spectral
