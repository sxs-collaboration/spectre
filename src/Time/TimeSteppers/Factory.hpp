// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/Cerk3.hpp"
#include "Time/TimeSteppers/Cerk4.hpp"
#include "Time/TimeSteppers/Cerk5.hpp"
#include "Time/TimeSteppers/ClassicalRungeKutta4.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {
/// Typelist of available TimeSteppers
using time_steppers =
    tmpl::list<TimeSteppers::AdamsBashforthN, TimeSteppers::Cerk3,
               TimeSteppers::Cerk4, TimeSteppers::Cerk5,
               TimeSteppers::ClassicalRungeKutta4, TimeSteppers::DormandPrince5,
               TimeSteppers::Heun2, TimeSteppers::RungeKutta3>;

/// Typelist of available LtsTimeSteppers
using lts_time_steppers = tmpl::list<TimeSteppers::AdamsBashforthN>;
}  // namespace Triggers
