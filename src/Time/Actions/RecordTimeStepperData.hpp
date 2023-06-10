// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class TimeStepId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace record_time_stepper_data_detail {
template <typename System, typename VariablesTag, typename DbTags>
void record_one_variables(const gsl::not_null<db::DataBox<DbTags>*> box) {
  using dt_variables_tag = db::add_tag_prefix<Tags::dt, VariablesTag>;
  using history_tag = Tags::HistoryEvolvedVariables<VariablesTag>;

  db::mutate<history_tag>(
      [](const gsl::not_null<typename history_tag::type*> history,
         const TimeStepId& time_step_id,
         const typename VariablesTag::type& vars,
         const typename dt_variables_tag::type& dt_vars) {
        history->insert(time_step_id, vars, dt_vars);
      },
      box, db::get<Tags::TimeStepId>(*box), db::get<VariablesTag>(*box),
      db::get<dt_variables_tag>(*box));
}
}  // namespace record_time_stepper_data_detail

/// Records the variables and their time derivatives in the time stepper
/// history.
///
/// \note this is a free function version of `Actions::RecordTimeStepperData`.
/// This free function alternative permits the inclusion of the time step
/// procedure in the middle of another action.
template <typename System, typename DbTags>
void record_time_stepper_data(const gsl::not_null<db::DataBox<DbTags>*> box) {
  if constexpr (tt::is_a_v<tmpl::list, typename System::variables_tag>) {
    // The system has multiple evolved variables, probably because
    // there is a mixture of real and complex values or similar.  Step
    // all of them.
    tmpl::for_each<typename System::variables_tag>([&](auto tag) {
      record_time_stepper_data_detail::record_one_variables<
          System, tmpl::type_from<decltype(tag)>>(box);
    });
  } else {
    record_time_stepper_data_detail::record_one_variables<
        System, typename System::variables_tag>(box);
  }
}

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Records the variables and their time derivatives in the
/// time stepper history.
///
/// With `dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>`:
///
/// Uses:
/// - GlobalCache: nothing
/// - DataBox:
///   - System::variables_tag
///   - dt_variables_tag
///   - Tags::HistoryEvolvedVariables<variables_tag>
///   - Tags::TimeStepId
///
/// DataBox changes:
/// - Tags::HistoryEvolvedVariables<variables_tag>
template <typename System>
struct RecordTimeStepperData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    record_time_stepper_data<System>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
