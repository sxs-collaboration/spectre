// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

namespace Convergence {

/*!
 * \brief The reason the algorithm has converged.
 *
 * \see Convergence::Criteria
 */
enum class Reason { MaxIterations, AbsoluteResidual, RelativeResidual };

std::ostream& operator<<(std::ostream& os, const Reason& reason) noexcept;

}  // namespace Convergence
