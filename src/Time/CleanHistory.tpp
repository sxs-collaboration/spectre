// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/CleanHistory.hpp"

#include "Time/History.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

template <typename System, template <typename> typename CacheTagPrefix,
          typename... VariablesTags>
void CleanHistory<System, CacheTagPrefix, tmpl::list<VariablesTags...>>::apply(
    const gsl::not_null<
        TimeSteppers::History<typename VariablesTags::type>*>... histories,
    const TimeStepper& time_stepper) {
  expand_pack((time_stepper.clean_history(histories), 0)...);
}
