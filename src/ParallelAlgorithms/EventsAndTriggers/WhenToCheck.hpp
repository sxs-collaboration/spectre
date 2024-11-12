// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines enum class Triggers::WhenToCheck.

#pragma once

#include <cstdint>
#include <iosfwd>

namespace Triggers {

/// \ingroup EventsAndTriggersGroup
/// \brief Frequency at which Events and Triggers are checked
enum class WhenToCheck : uint8_t {
  AtIterations, /**< checked at iterations e.g. of an elliptic solve */
  AtSlabs,      /**< checked at time Slab boundaries */
  AtSteps,      /**< checked at time step boundaries */
};

/// Output operator for a WhenToCheck.
std::ostream& operator<<(std::ostream& os, const WhenToCheck& when_to_check);
}  // namespace Triggers
