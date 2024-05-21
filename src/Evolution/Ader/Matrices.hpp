// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>

/// \cond
class Matrix;
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace ader::dg {
/*!
 * \brief Computes the matrix applied to the spacetime predictor volume
 * contributions to give the next initial guess in the iteration.
 *
 * Specifically, the returned matrix is:
 *
 * \f{align}{
 * S_{a_0 b_0}=\left(\ell_{a_0}(1)\ell_{c_0}(1) - w_{c_0}
 * \left(\partial_\tau\ell_{a_0}(\tau)\right)\rvert_{c_0}
 * \right)^{-1}M_{c_{0}b_{0}}
 * \f}
 *
 * where \f$M_{a_0 b_0}\f$ is the temporal mass matrix.
 */
template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
const Matrix& predictor_inverse_temporal_matrix(size_t num_points);
}  // namespace ader::dg
