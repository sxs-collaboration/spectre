// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action UpdateU

#pragma once

#include <optional>
#include <tuple>
#include <utility>  // IWYU pragma: keep // for std::move

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "Time/Time.hpp" // for TimeDelta

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare TimeDelta
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

/// Perform variable updates for one substep for a substep method, or one step
/// for an LMM method.
///
/// \note This is a free function version of `Actions::UpdateU`. This free
/// function alternative permits the inclusion of the time step procedure in
/// the middle of another action.
template <typename System, typename VariablesTag = NoSuchType, typename DbTags>
void update_u(const gsl::not_null<db::DataBox<DbTags>*> box) {
  using variables_tag =
      tmpl::conditional_t<std::is_same_v<VariablesTag, NoSuchType>,
                          typename System::variables_tag, VariablesTag>;
  using history_tag = Tags::HistoryEvolvedVariables<variables_tag>;
  bool is_using_error_control = false;
  if constexpr (db::tag_is_retrievable_v<Tags::IsUsingTimeSteppingErrorControl,
                                         db::DataBox<DbTags>>) {
    is_using_error_control =
        db::get<Tags::IsUsingTimeSteppingErrorControl>(*box);
  }
  if (is_using_error_control) {
    using error_tag = ::Tags::StepperError<variables_tag>;
    using previous_error_tag = ::Tags::PreviousStepperError<variables_tag>;
    if constexpr (tmpl::list_contains_v<DbTags, error_tag>) {
      db::mutate<Tags::StepperErrorUpdated, variables_tag, error_tag,
                 previous_error_tag, history_tag>(
          box,
          [](const gsl::not_null<bool*> stepper_error_updated,
             const gsl::not_null<typename variables_tag::type*> vars,
             const gsl::not_null<typename error_tag::type*> error,
             const gsl::not_null<typename previous_error_tag::type*>
                 previous_error,
             const gsl::not_null<typename history_tag::type*> history,
             const ::TimeDelta& time_step, const auto& time_stepper) {
            using std::swap;
            // We need to make sure *previous_error has the correct
            // size.  We don't care about the value, but it could be
            // several types so we can't just call ->initialize() or
            // something.
            //
            // We are not required to preserve the old value, because
            // the errors will only be used after a successful error
            // update.
            *previous_error = *vars;
            swap(*error, *previous_error);
            *stepper_error_updated = time_stepper.update_u(
                vars, make_not_null(&*error), history, time_step);
            if (not *stepper_error_updated) {
              swap(*error, *previous_error);
            }
          },
          db::get<Tags::TimeStep>(*box), db::get<Tags::TimeStepper<>>(*box));
    } else {
      ERROR(
          "Cannot update the stepper error measure -- "
          "`::Tags::StepperError<VariablesTag>` is not present in the box.");
    }
  } else {
    db::mutate<variables_tag, history_tag>(
        box,
        [](const gsl::not_null<typename variables_tag::type*> vars,
           const gsl::not_null<typename history_tag::type*> history,
           const ::TimeDelta& time_step, const auto& time_stepper) {
          time_stepper.update_u(vars, history, time_step);
        },
        db::get<Tags::TimeStep>(*box), db::get<Tags::TimeStepper<>>(*box));
  }
}

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Perform variable updates for one substep
///
/// Uses:
/// - DataBox:
///   - variables_tag (either the provided `VariablesTag` or the
///   `system::variables_tag` if none is provided)
///   - Tags::HistoryEvolvedVariables<variables_tag>
///   - Tags::TimeStep
///   - Tags::TimeStepper<>
///   - Tags::IsUsingTimeSteppingErrorControl
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - variables_tag
///   - Tags::HistoryEvolvedVariables<variables_tag>
template <typename VariablesTag = NoSuchType>
struct UpdateU {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    update_u<typename Metavariables::system, VariablesTag>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
