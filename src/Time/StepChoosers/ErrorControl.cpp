// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepChoosers/ErrorControl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <type_traits>

#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorEstimate.hpp"

namespace StepChoosers::ErrorControl_detail {
template <typename StepChooserUse>
std::optional<double> goal_from_variable(
    const std::array<std::optional<StepperErrorEstimate>, 2>& errors,
    const double min_factor, const double max_factor,
    const double safety_factor) {
  // Do not request that the step size be changed if there isn't a new error
  // estimate
  if (not errors[1].has_value()) {
    return {};
  }
  if (std::is_same_v<StepChooserUse, ::StepChooserUse::LtsStep> or
      not errors[0].has_value() or errors[0]->order != errors[1]->order) {
    return errors[1]->step_size.value() *
           std::clamp(safety_factor *
                          pow(1.0 / std::max(errors[1]->step_error(), 1e-14),
                              1.0 / static_cast<double>(errors[1]->order + 1)),
                      min_factor, max_factor);
  } else {
    // From simple advice from Numerical Recipes 17.2.1 regarding a heuristic
    // for PI step control.
    const double alpha_factor = 0.7 / static_cast<double>(errors[1]->order + 1);
    const double beta_factor = 0.4 / static_cast<double>(errors[0]->order + 1);
    return errors[1]->step_size.value() *
           std::clamp(
               safety_factor *
                   pow(1.0 / std::max(errors[1]->step_error(), 1e-14),
                       alpha_factor) *
                   pow(std::max(errors[0]->step_error(), 1e-14), beta_factor),
               min_factor, max_factor);
  }
}

template std::optional<double> goal_from_variable<StepChooserUse::Slab>(
    const std::array<std::optional<StepperErrorEstimate>, 2>& errors,
    double min_factor, double max_factor, double safety_factor);
template std::optional<double> goal_from_variable<StepChooserUse::LtsStep>(
    const std::array<std::optional<StepperErrorEstimate>, 2>& errors,
    double min_factor, double max_factor, double safety_factor);
}  // namespace StepChoosers::ErrorControl_detail
