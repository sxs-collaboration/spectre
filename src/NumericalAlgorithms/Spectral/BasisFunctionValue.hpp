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
 * \brief Compute the function values of the basis function \f$\Phi_k(x)\f$
 * (zero-indexed).
 */
template <Basis BasisType, typename T>
T compute_basis_function_value(size_t k, const T& x);

/*!
 * \brief Compute the function values of the basis function \f$\Phi^m_k(x)\f$
 * (zero-indexed).
 */
template <Basis BasisType, typename T>
T compute_basis_function_value(size_t k, size_t m, const T& x);
}  // namespace Spectral
