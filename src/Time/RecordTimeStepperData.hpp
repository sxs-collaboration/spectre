// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct HistoryEvolvedVariables;
struct TimeStepId;
}  // namespace Tags
/// \endcond

/// \ingroup TimeGroup
/// Records the variables and their time derivatives in the time stepper
/// history.
/// @{
template <typename System,
          typename = tmpl::conditional_t<
              tt::is_a_v<tmpl::list, typename System::variables_tag>,
              typename System::variables_tag,
              tmpl::list<typename System::variables_tag>>>
struct RecordTimeStepperData;

template <typename System, typename... VariablesTags>
struct RecordTimeStepperData<System, tmpl::list<VariablesTags...>> {
  using return_tags =
      tmpl::list<Tags::HistoryEvolvedVariables<VariablesTags>...>;
  using argument_tags =
      tmpl::list<Tags::TimeStepId, VariablesTags...,
                 db::add_tag_prefix<Tags::dt, VariablesTags>...>;

  static void apply(
      const gsl::not_null<
          TimeSteppers::History<typename VariablesTags::type>*>... histories,
      const TimeStepId& time_step_id,
      const typename VariablesTags::type&... vars,
      const typename db::add_tag_prefix<Tags::dt,
                                        VariablesTags>::type&... dt_vars) {
    expand_pack((histories->insert(time_step_id, vars, dt_vars), 0)...);
  }
};
/// @}
