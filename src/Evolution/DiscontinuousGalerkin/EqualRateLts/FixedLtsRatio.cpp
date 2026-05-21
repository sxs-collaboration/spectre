// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/FixedLtsRatio.hpp"

#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Time/CollectStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"

namespace evolution::dg::StepChoosers {
FixedLtsRatio::FixedLtsRatio(
    std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>
        step_choosers)
    : step_choosers_(std::move(step_choosers)) {}

bool FixedLtsRatio::uses_local_data() const { return true; }
bool FixedLtsRatio::can_be_delayed() const { return true; }

std::unordered_map<std::type_index, StepperErrorTolerances>
FixedLtsRatio::tolerances() const {
  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{};
  for (const auto& step_chooser : step_choosers_) {
    collect_stepper_error_tolerances(&tolerances, *step_chooser);
  }
  return tolerances;
}

void FixedLtsRatio::pup(PUP::er& p) {
  StepChooser<StepChooserUse::Slab>::pup(p);
  p | step_choosers_;
}

PUP::able::PUP_ID FixedLtsRatio::my_PUP_ID = 0;  // NOLINT
}  // namespace evolution::dg::StepChoosers
