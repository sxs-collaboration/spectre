// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the scalar advection equation.
 *
 * \f{align*}
 * \partial_t U + \nabla \cdot (v U) = 0
 * \f}
 *
 * Since the ScalarAdvection system is only used for testing limiters in the
 * current implementation, the velocity field \f$v\f$ is fixed throughout time.
 */
namespace ScalarAdvection {
template <size_t Dim>
struct System {};
}  // namespace ScalarAdvection
