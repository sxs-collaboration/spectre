// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
class DataVector;
namespace Spectral {
enum class Basis : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {
/*!
 * \brief Compute the inverse of the weight function \f$w(x)\f$ w.r.t. which
 * the basis functions are orthogonal. See the description of
 * `quadrature_weights(size_t)` for details.
 * This is arbitrarily set to 1 for FiniteDifference basis, to integrate
 * using the midpoint method (see `quadrature_weights (size_t)` for details).
 */
template <Basis>
DataVector compute_inverse_weight_function_values(const DataVector&);
}  // namespace Spectral
