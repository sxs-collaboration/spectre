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
#include "Utilities/NoSuchType.hpp"
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

/// Records the variables and their time derivatives in the time stepper
/// history, and copies `variables_tag` to `Tags::RollbackValue<variables_tag>`
/// if the latter is present in the passed `DataBox`.
///
/// \note this is a free function version of `Actions::RecordTimeStepperData`.
/// This free function alternative permits the inclusion of the time step
/// procedure in the middle of another action.
template <typename System, typename VariablesTag = NoSuchType, typename DbTags>
void record_time_stepper_data(const gsl::not_null<db::DataBox<DbTags>*> box) {
  using variables_tag =
      tmpl::conditional_t<std::is_same_v<VariablesTag, NoSuchType>,
                          typename System::variables_tag, VariablesTag>;
  using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;
  using history_tag = Tags::HistoryEvolvedVariables<variables_tag>;

  db::mutate<history_tag>(
      box,
      [](const gsl::not_null<typename history_tag::type*> history,
         const TimeStepId& time_step_id,
         const typename dt_variables_tag::type& dt_vars) {
        history->insert(time_step_id, dt_vars);
      },
      db::get<Tags::TimeStepId>(*box), db::get<dt_variables_tag>(*box));

  // Not all executables perform rollbacks, so only save the data if
  // something uses it.  (In which case it is that code's job to make
  // sure it's in the DataBox.)
  using rollback_tag = Tags::RollbackValue<variables_tag>;
  if constexpr (db::tag_is_retrievable_v<rollback_tag, db::DataBox<DbTags>>) {
    db::mutate<rollback_tag>(
        box,
        [](const gsl::not_null<typename rollback_tag::type*> rollback_value,
           const typename variables_tag::type& vars) {
          *rollback_value = vars;
        },
        db::get<variables_tag>(*box));
  }
}

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Records the variables and their time derivatives in the
/// time stepper history, and copies `variables_tag` to
/// `Tags::RollbackValue<variables_tag>` if the latter is present in the
/// `DataBox`.
///
/// With `dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>`:
///
/// Uses:
/// - GlobalCache: nothing
/// - DataBox:
///   - variables_tag (either the provided `VariablesTag` or the
///   `system::variables_tag` if none is provided)
///   - dt_variables_tag
///   - Tags::HistoryEvolvedVariables<variables_tag>
///   - Tags::TimeStepId
///
/// DataBox changes:
/// - Tags::HistoryEvolvedVariables<variables_tag>
/// - Tags::RollbackValue<variables_tag> if present
template <typename VariablesTag = NoSuchType>
struct RecordTimeStepperData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    record_time_stepper_data<typename Metavariables::system, VariablesTag>(
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
