// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"

/// \cond
class DataVector;
class Matrix;
namespace Spectral {
enum class Quadrature : uint8_t;
enum class Parity : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {

/*!
 * \ingroup SpectralGroup
 *
 * \brief A collection of helper functions for the radial functions used in
 * Zernike polyomials
 *
 */
template <size_t Dim>
class Zernike {
 public:
  /*!
   * \brief Value of the basis function \f$\Phi^m_n(\xi) = R^m_n(r)\f$,
   * where \f$r \equiv \frac{1}{2} (\xi + 1)\f$, implemented from
   * \cite Matsushima1995
   */
  template <typename T>
  static T basis_function_value(size_t n, size_t m, const T& xi);

  /*!
   * \brief Collocation points \f${x_i}\f$ and quadrature weights \f${w_i}\f$
   */
  static std::pair<DataVector, DataVector>
  compute_collocation_points_and_weights(size_t num_points);

  /*!
   * \brief Matrix \f$D_{i,j}\f$ used to obtain the first derivative for a
   * given parity.
   *
   * Due to the clustering of Zernike collocation toward the upper side, the
   * generic implementation of derivatives with barycentric weights yields
   * large errors. By utilizing the fact that the Zernike bases' \f$m\f$
   * corresponds to parity of representable functions, we can extend the
   * function to negative \f$r\f$ before forming the matrix, greatly improving
   * accuracy.
   */
  static Matrix differentiation_matrix(size_t num_points, Parity parity);

  /*!
   * \brief %Matrix used to interpolate to the \p target_points.
   *
   * Due to the clustering of Zernike collocation toward the upper side, the
   * generic barycentric interpolation yields large errors. By utilizing the
   * fact that the Zernike bases' \f$m\f$ corresponds to parity of
   * representable functions, we extend the function to negative \f$r\f$ before
   * forming the matrix, greatly improving accuracy.
   *
   * The returned matrix \f$M\f$ has \f$n_\mathrm{target}\f$ rows and
   * \p num_points columns, so that \f$f_\mathrm{target} = M
   * f_\mathrm{source}\f$.
   *
   * This should only be used for ZernikeB1. To interpolate ZernikeB2 or
   * ZernikeB3, look at Irregular or Cardinal.
   */
  template <typename T>
  static Matrix interpolation_matrix(size_t num_points, const T& target_points,
                                     Parity parity);
};
}  // namespace Spectral
