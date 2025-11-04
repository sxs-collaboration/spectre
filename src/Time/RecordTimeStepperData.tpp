// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/RecordTimeStepperData.hpp"

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

template <typename System, typename... VariablesTags>
void RecordTimeStepperData<System, tmpl::list<VariablesTags...>>::apply(
    const gsl::not_null<
        TimeSteppers::History<typename VariablesTags::type>*>... histories,
    const TimeStepId& time_step_id, const typename VariablesTags::type&... vars,
    const typename db::add_tag_prefix<Tags::dt,
                                      VariablesTags>::type&... dt_vars) {
  expand_pack((histories->insert(time_step_id, vars, dt_vars), 0)...);
}
