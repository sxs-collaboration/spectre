// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/FixedLtsRatio.hpp"

#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>
#include <vector>

#include "Time/StepChoosers/StepChooser.hpp"

namespace StepChoosers {
FixedLtsRatio::FixedLtsRatio(
    std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>
        step_choosers)
    : step_choosers_(std::move(step_choosers)) {}

bool FixedLtsRatio::uses_local_data() const { return true; }
bool FixedLtsRatio::can_be_delayed() const { return true; }

void FixedLtsRatio::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  p | step_choosers_;
}

PUP::able::PUP_ID FixedLtsRatio::my_PUP_ID = 0;  // NOLINT
}  // namespace StepChoosers
