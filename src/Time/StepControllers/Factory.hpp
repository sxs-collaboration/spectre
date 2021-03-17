// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/StepControllers/BinaryFraction.hpp"
#include "Utilities/TMPL.hpp"

namespace StepControllers {
/// Typelist of standard StepControllers
using standard_step_controllers = tmpl::list<BinaryFraction>;
}  // namespace Triggers
