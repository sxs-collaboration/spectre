// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Documents the `sgb` namespace

#pragma once

/*!
 * \brief Holds items related to Einstein-scalar-Gauss-Bonnet gravity.
 *
 * Einstein-scalar-Gauss-Bonnet is a modified gravity theory featuring a real
 * scalar field nonminimally coupled to the metric. In this code we will follow
 * the conventions of \cite Nee2024bur , and write the action as
 * \begin{equation}
 *  S = \int_\Omega d^4 x \, \sqrt{-g} \biggl\{ \frac{\mathcal{R}}{16 \pi G}
 *    - \frac{1}{2} \bigl( \nabla_\mu \Psi \bigr) \bigl( \nabla^\mu \Psi \bigr)
 *    + \ell^2 F[\Psi] \mathcal{G} \biggr\}
 * \end{equation}
 * where \f$\mathcal{R}\f$ is the Ricci scalar, \f$G\f$ is the Newton's,
 * constant, \f$\Psi\f$ is the real scalar field, \f$F[\Psi]\f$ is its coupling
 * function, and \f$\mathcal{G}\f$ is the Gauss-Bonnet invariant.
 */
namespace ScalarTensor::sgb {}
