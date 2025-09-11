// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Imex/CleanHistory.hpp"

#include "DataStructures/Variables.hpp"
#include "Time/History.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace imex {
template <typename System, typename... ImplicitSectors>
void CleanHistory<System, tmpl::list<ImplicitSectors...>>::apply(
    const gsl::not_null<TimeSteppers::History<
        Variables<typename ImplicitSectors::tensors>>*>... histories,
    const ImexTimeStepper& time_stepper) {
  expand_pack((time_stepper.clean_history(histories), 0)...);
}
}  // namespace imex
