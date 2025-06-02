// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace Spectral {
enum class Basis : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Spectral {
/*!
 * \brief Compute the normalization square of the basis function \f$\Phi_k\f$
 * (zero-indexed), i.e. the weighted definite integral over its square.
 */
template <Basis BasisType>
double compute_basis_function_normalization_square(size_t k);
}  // namespace Spectral
