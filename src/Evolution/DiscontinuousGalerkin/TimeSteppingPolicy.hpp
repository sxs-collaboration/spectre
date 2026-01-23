// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <iosfwd>

namespace evolution::dg {
/// \brief Treatment of time stepping across an element boundary.
///
/// \details Indicates how boundary corrections should be handled for
/// local time-stepping across a mortar.  This controls the
/// communication algorithm and data structures used.
///
/// \note The code controlling the time step will not directly check
/// this policy, but must be configured to make choices consistent
/// with it.
enum class TimeSteppingPolicy : uint8_t {
  /// Default value is uninitialized.
  Uninitialized,
  /// The elements on the two sides of the mortar must have the same
  /// step size.  Communicated boundary corrections will be added
  /// directly to the volume RHS calculation, and algorithms that are
  /// not local-time-stepping aware (such as subcell) can be used at
  /// this boundary.  This is the only valid policy in a global
  /// time-stepping evolution.
  EqualRate,
  /// The elements on the two sides of the mortar may have different
  /// step sizes.  A flux-conservative boundary integral will be
  /// performed to couple the elements.
  Conservative,
};

/// Output operator for a TimeSteppingPolicy.
std::ostream& operator<<(std::ostream& os, TimeSteppingPolicy value);
}  // namespace evolution::dg
