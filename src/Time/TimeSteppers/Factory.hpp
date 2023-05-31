// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/ClassicalRungeKutta4.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/Rk3Owren.hpp"
#include "Time/TimeSteppers/Rk4Owren.hpp"
#include "Time/TimeSteppers/Rk5Owren.hpp"
#include "Time/TimeSteppers/Rk5Tsitouras.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {
/// Typelist of available TimeSteppers
using time_steppers =
    tmpl::list<TimeSteppers::AdamsBashforth, TimeSteppers::AdamsMoultonPc,
               TimeSteppers::ClassicalRungeKutta4, TimeSteppers::DormandPrince5,
               TimeSteppers::Heun2, TimeSteppers::Rk3HesthavenSsp,
               TimeSteppers::Rk3Owren, TimeSteppers::Rk4Owren,
               TimeSteppers::Rk5Owren, TimeSteppers::Rk5Tsitouras>;

/// Typelist of available LtsTimeSteppers
using lts_time_steppers = tmpl::list<TimeSteppers::AdamsBashforth>;
}  // namespace Triggers
