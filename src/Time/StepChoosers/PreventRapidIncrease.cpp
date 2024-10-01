// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/PreventRapidIncrease.hpp"

#include <pup.h>

#include "Time/StepChoosers/StepChooser.hpp"

namespace StepChoosers {
bool PreventRapidIncrease::uses_local_data() const { return false; }

bool PreventRapidIncrease::can_be_delayed() const { return true; }

void PreventRapidIncrease::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  StepChooser<StepChooserUse::LtsStep>::pup(p);
}

PUP::able::PUP_ID PreventRapidIncrease::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers
