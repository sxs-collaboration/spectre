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
 * \brief %Matrix used to interpolate to the \p target_points.
 *
 * \warning For each target point located outside of the logical coordinate
 * interval covered by `BasisType` (often \f$[-1,1]\f$), the resulting matrix
 * performs polynomial extrapolation instead of interpolation. The extrapolation
 * will be correct but may suffer from reduced accuracy, especially for
 * higher-order polynomials (i.e., larger values of `num_points`).
 *
 * \param num_points The number of collocation points
 * \param target_points The points to interpolate to
 */
template <Basis BasisType, Quadrature QuadratureType, typename T>
Matrix interpolation_matrix(size_t num_points, const T& target_points);

/*!
 * \brief Interpolation matrix to the \p target_points for a one-dimensional
 * mesh.
 *
 * \see interpolation_matrix(size_t, const T&)
 */
template <typename T>
Matrix interpolation_matrix(const Mesh<1>& mesh, const T& target_points);
}  // namespace Spectral
