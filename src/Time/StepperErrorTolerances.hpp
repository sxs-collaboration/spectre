// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>
#include <limits>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup TimeGroup
/// Tolerances used for time step error control
struct StepperErrorTolerances {
  enum class Estimates { None, StepperOrder, AllOrders };

  /// Which estimates the time stepper should generate.
  Estimates estimates = Estimates::None;
  double absolute = std::numeric_limits<double>::signaling_NaN();
  double relative = std::numeric_limits<double>::signaling_NaN();

  void pup(PUP::er& p);
};

bool operator==(const StepperErrorTolerances& a,
                const StepperErrorTolerances& b);
bool operator!=(const StepperErrorTolerances& a,
                const StepperErrorTolerances& b);

std::ostream& operator<<(std::ostream& os,
                         const StepperErrorTolerances& tolerances);
