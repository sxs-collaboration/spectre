// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

namespace dg {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief The DG formulation to use
 *
 * - The `StrongInertial` formulation is also known as the integrate then
 *   transform formulation. The "Inertial" part of the name refers to the fact
 *   that the integration is done over the physical/inertial coordinates, while
 *   the "strong" part refers to the fact that the boundary correction terms are
 *   zero if the solution is continuous at the interfaces.
 *   See \cite Teukolsky2015ega for an overview.
 * - The `WeakInertial` formulation is also known as the integrate then
 *   transform formulation. The "Inertial" part of the name refers to the fact
 *   that the integration is done over the physical/inertial coordinates, while
 *   the "weak" part refers to the fact that the boundary correction terms are
 *   non-zero even if the solution is continuous at the interfaces.
 *   See \cite Teukolsky2015ega for an overview.
 */
enum class Formulation { StrongInertial, WeakInertial };

std::ostream& operator<<(std::ostream& os, Formulation t) noexcept;
}  // namespace dg
