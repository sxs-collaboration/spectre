// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/ClassicalRungeKutta4.hpp"
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/Heun2.hpp"
#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"
#include "Time/TimeSteppers/Rk3Kennedy.hpp"
#include "Time/TimeSteppers/Rk3Owren.hpp"
#include "Time/TimeSteppers/Rk3Pareschi.hpp"
#include "Time/TimeSteppers/Rk4Kennedy.hpp"
#include "Time/TimeSteppers/Rk4Owren.hpp"
#include "Time/TimeSteppers/Rk5Owren.hpp"
#include "Time/TimeSteppers/Rk5Tsitouras.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {
/// Typelist of available TimeSteppers
using time_steppers =
    tmpl::list<AdamsBashforth, AdamsMoultonPc<false>, AdamsMoultonPc<true>,
               ClassicalRungeKutta4, DormandPrince5, Heun2, Rk3HesthavenSsp,
               Rk3Kennedy, Rk3Owren, Rk3Pareschi, Rk4Kennedy, Rk4Owren,
               Rk5Owren, Rk5Tsitouras>;

/// Typelist of available LtsTimeSteppers
using lts_time_steppers = tmpl::list<AdamsBashforth>;

/// Typelist of available ImexTimeSteppers
using imex_time_steppers =
    tmpl::list<Heun2, Rk3Kennedy, Rk3Pareschi, Rk4Kennedy>;

/// Typelist of TimeSteppers whose substep times are strictly increasing
using increasing_substep_time_steppers =
    tmpl::list<AdamsBashforth, Rk3Owren, Rk4Owren>;
}  // namespace TimeSteppers
