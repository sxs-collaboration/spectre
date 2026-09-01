// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
class Matrix;
namespace Spectral {
enum class Parity : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {

/*!
 * \ingroup SpectralGroup
 *
 * \brief A collection of helper functions for the half-Fourier spectral basis
 *
 * \details The half-Fourier basis represents functions on the interval
 * \f$\phi \in [0, \pi)\f$ using \f$N\f$ equispaced interior collocation
 * points \f$\phi_j = (j + \tfrac{1}{2})\pi/N\f$ for \f$j = 0, \ldots, N-1\f$.
 *
 * Functions of even parity under \f$\phi \to -\phi\f$ (i.e. those satisfying
 * \f$f(-\phi)=f(\phi)\f$) are expanded in cosines:
 * \f[
 * f(\phi) = \sum_{n=0}^{N-1} a_n \cos(n\phi)
 * \f]
 *
 * Functions of odd parity under \f$\phi \to -\phi\f$ (i.e. those satisfying
 * \f$f(-\phi)=-f(\phi)\f$) are expanded in sines:
 * \f[
 * f(\phi) = \sum_{n=1}^{N} b_n \sin(n\phi)
 * \f]
 *
 * The derivative \f$\partial / \partial \phi\f$ maps even-parity functions to
 * odd-parity functions and vice versa.
 *
 * This basis is intended for use in the Cartoon method for axisymmetric
 * problems on a cylinder, where the azimuthal direction covers only half a
 * circle due to the reflection symmetry, and the parity boundary conditions
 * are internal to the spectral representation.
 */
class HalfFourier {
 public:
  /*!
   * \brief Collocation points \f$\{\phi_j\}\f$
   *
   * \details The collocation points on the interval \f$(0, \pi)\f$ are given
   * by
   * \f[
   * \phi_j = \frac{(j + \tfrac{1}{2})\pi}{N}
   * \f]
   */
  static DataVector collocation_points(size_t num_points);

  /*!
   * \brief Quadrature weights \f$\{w_j\}\f$
   *
   * \details The quadrature weights are uniform:
   * \f[
   * w_j = \frac{\pi}{N}
   * \f]
   */
  static DataVector quadrature_weights(size_t num_points);

  /*!
   * \brief Differentiation matrix \f$D^{\text{even}}_{ij}\f$ for even-parity
   * functions
   *
   * \details Maps an even-parity function (expanded in cosines) to its
   * derivative, which is an odd-parity function (expanded in sines).
   * Explicitly:
   * \f[
   * D^{\text{even}}_{ij} = \frac{2}{N} \sum_{n=1}^{N-1}
   * (-n) \sin(n\phi_i) \cos(n\phi_j)
   * \f]
   */
  static Matrix even_differentiation_matrix(size_t num_points);

  /*!
   * \brief Differentiation matrix \f$D^{\text{odd}}_{ij}\f$ for odd-parity
   * functions
   *
   * \details Maps an odd-parity function (expanded in sines) to its
   * derivative, which is an even-parity function (expanded in cosines).
   * Explicitly:
   * \f[
   * D^{\text{odd}}_{ij} = \frac{2}{N} \sum_{n=1}^{N-1}
   * n \cos(n\phi_i) \sin(n\phi_j)
   * \f]
   *
   * Note that \f$D^{\text{even}} = -(D^{\text{odd}})^T\f$.
   */
  static Matrix odd_differentiation_matrix(size_t num_points);

  /*!
   * \brief Interpolation matrix for even-parity functions to
   * \p target_points.
   *
   * \details Using the discrete cosine transform (DCT-II) representation, the
   * interpolation weights at a target point \f$x\f$ are:
   * \f[
   * I^{\text{even}}_j(x) = \frac{1}{N}\left[1 + 2\sum_{n=1}^{N-1}
   * \cos(nx)\cos(n\phi_j)\right]
   * \f]
   */
  template <typename T>
  static Matrix even_interpolation_matrix(size_t num_points,
                                          const T& target_points);

  /*!
   * \brief Interpolation matrix for odd-parity functions to
   * \p target_points.
   *
   * \details Using the discrete sine transform (DST-II) representation, the
   * interpolation weights at a target point \f$x\f$ are:
   * \f[
   * I^{\text{odd}}_j(x) = \frac{2}{N}\sum_{n=1}^{N-1}
   * \sin(nx)\sin(n\phi_j) + \frac{1}{N}\sin(Nx)\sin(N\phi_j)
   * \f]
   * where the Nyquist mode \f$n=N\f$ carries half the weight of the other
   * modes.
   */
  template <typename T>
  static Matrix odd_interpolation_matrix(size_t num_points,
                                         const T& target_points);

  /*!
   * \brief %Matrix used to interpolate to the \p target_points for a given
   * \p parity.
   *
   * \details Dispatches to `even_interpolation_matrix` when \p parity is
   * `Parity::Even` and to `odd_interpolation_matrix` when it is `Parity::Odd`.
   * This mirrors the interface of `Zernike<1>::interpolation_matrix`.
   */
  template <typename T>
  static Matrix interpolation_matrix(size_t num_points, const T& target_points,
                                     Parity parity);
};
}  // namespace Spectral
