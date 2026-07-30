// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ScalarWave::Solutions and SecondOrderScalarWave::Solutions

#pragma once

namespace ScalarWave {
/*!
 * \ingroup AnalyticSolutionsGroup
 * \brief Holds analytic solutions to the first-order reduction of the
 * Euclidean wave equation.
 *
 * The first-order system evolves \f$\Psi\f$, \f$\Pi\f$, and \f$\Phi_j\f$:
 * \f{align*}
 * \partial_t \Psi &= -\Pi, \\
 * \partial_t \Pi &= -\delta^{ij}\partial_i\Phi_j, \\
 * \partial_t \Phi_j &= -\partial_j\Pi.
 * \f}
 */
namespace Solutions {}
}  // namespace ScalarWave

/*!
 * \ingroup AnalyticSolutionsGroup
 * \brief Holds analytic solutions to the second-order-in-space formulation of
 * the Euclidean wave equation.
 *
 * The second-order-in-space system evolves \f$\Psi\f$ and \f$\Pi\f$, while
 * \f$\Phi_j\f$ is auxiliary:
 * \f{align*}
 * \partial_t \Psi &= -\Pi, \\
 * \partial_t \Pi &= -\delta^{ij}\partial_i\Phi_j, \\
 * \Phi_j &= \partial_j\Psi.
 * \f}
 */
namespace SecondOrderScalarWave::Solutions {}
