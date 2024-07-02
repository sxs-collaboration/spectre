// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup TimeGroup
/// Tolerances used for time step error control
struct StepperErrorTolerances {
  double absolute = std::numeric_limits<double>::signaling_NaN();
  double relative = std::numeric_limits<double>::signaling_NaN();

  void pup(PUP::er& p);
};

bool operator==(const StepperErrorTolerances& a,
                const StepperErrorTolerances& b);
bool operator!=(const StepperErrorTolerances& a,
                const StepperErrorTolerances& b);
