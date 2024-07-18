// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action UpdateU

#pragma once

#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/SelfStart.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct IsUsingTimeSteppingErrorControl;
template <typename Tag>
struct StepperErrorTolerances;
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
/// \endcond

namespace update_u_detail {
template <typename System, typename VariablesTag, typename DbTags>
void update_one_variables(const gsl::not_null<db::DataBox<DbTags>*> box) {
  using history_tag = Tags::HistoryEvolvedVariables<VariablesTag>;
  bool is_using_error_control = false;
  if constexpr (db::tag_is_retrievable_v<Tags::IsUsingTimeSteppingErrorControl,
                                         db::DataBox<DbTags>>) {
    is_using_error_control =
        db::get<Tags::IsUsingTimeSteppingErrorControl>(*box);
  }
  if (is_using_error_control) {
    using error_tag = ::Tags::StepperErrors<VariablesTag>;
    if constexpr (tmpl::list_contains_v<DbTags, error_tag>) {
      db::mutate<VariablesTag, error_tag>(
          [&](const gsl::not_null<typename VariablesTag::type*> vars,
              const gsl::not_null<typename error_tag::type*> errors,
              const typename history_tag::type& history,
              const ::TimeDelta& time_step, const TimeStepper& time_stepper,
              const std::optional<StepperErrorTolerances>& tolerances) {
            const auto error =
                time_stepper.update_u(vars, history, time_step, tolerances);
            if (error.has_value()) {
              // Save the previous errors, but not if this is a retry
              // of the same step.
              if ((*errors)[1]->step_time != error->step_time) {
                (*errors)[0] = (*errors)[1];
              }
              (*errors)[1].emplace(*error);
            }
          },
          box, db::get<history_tag>(*box), db::get<Tags::TimeStep>(*box),
          db::get<Tags::TimeStepper<TimeStepper>>(*box),
          db::get<Tags::StepperErrorTolerances<VariablesTag>>(*box));
    } else {
      ERROR(
          "Cannot update the stepper error measure -- "
          "`::Tags::StepperErrors<VariablesTag>` is not present in the box.");
    }
  } else {
    db::mutate<VariablesTag>(
        [](const gsl::not_null<typename VariablesTag::type*> vars,
           const typename history_tag::type& history,
           const ::TimeDelta& time_step, const TimeStepper& time_stepper) {
          time_stepper.update_u(vars, history, time_step);
        },
        box, db::get<history_tag>(*box), db::get<Tags::TimeStep>(*box),
        db::get<Tags::TimeStepper<TimeStepper>>(*box));
  }
}
}  // namespace update_u_detail

/// Perform variable updates for one substep for a substep method, or one step
/// for an LMM method.
///
/// \note This is a free function version of `Actions::UpdateU`. This free
/// function alternative permits the inclusion of the time step procedure in
/// the middle of another action.
template <typename System, typename DbTags>
void update_u(const gsl::not_null<db::DataBox<DbTags>*> box) {
  if (::SelfStart::step_unused(db::get<Tags::TimeStepId>(*box),
                               db::get<Tags::Next<Tags::TimeStepId>>(*box))) {
    return;
  }

  if constexpr (tt::is_a_v<tmpl::list, typename System::variables_tag>) {
    // The system has multiple evolved variables, probably because
    // there is a mixture of real and complex values or similar.  Step
    // all of them.
    tmpl::for_each<typename System::variables_tag>([&](auto tag) {
      update_u_detail::update_one_variables<System,
                                            tmpl::type_from<decltype(tag)>>(
          box);
    });
  } else {
    update_u_detail::update_one_variables<System,
                                          typename System::variables_tag>(box);
  }
}

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Perform variable updates for one substep
///
/// Uses:
/// - DataBox:
///   - system::variables_tag
///   - Tags::HistoryEvolvedVariables<variables_tag>
///   - Tags::TimeStep
///   - Tags::TimeStepper<TimeStepper>
///   - Tags::IsUsingTimeSteppingErrorControl
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - variables_tag
template <typename System>
struct UpdateU {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    update_u<System>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
