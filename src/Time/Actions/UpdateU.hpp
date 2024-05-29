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
#include "Time/SelfStart.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
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
          [](const gsl::not_null<typename VariablesTag::type*> vars,
             const gsl::not_null<typename error_tag::type*> errors,
             const typename history_tag::type& history,
             const ::TimeDelta& time_step, const TimeStepper& time_stepper) {
            using std::swap;
            bool new_entry = false;
            const auto current_step_time =
                history.back().time_step_id.step_time();
            if (not errors->back().has_value() or
                errors->back()->step_time != current_step_time) {
              swap((*errors)[0], (*errors)[1]);
              if (not errors->back().has_value()) {
                new_entry = true;
                errors->back().emplace();
                set_number_of_grid_points(make_not_null(&errors->back()->error),
                                          *vars);
              }
            }
            const bool stepper_error_updated = time_stepper.update_u(
                vars, make_not_null(&errors->back()->error), history,
                time_step);
            if (stepper_error_updated) {
              errors->back()->step_time = current_step_time;
              errors->back()->step_size = time_step;
              errors->back()->order = time_stepper.error_estimate_order();
            } else if (new_entry) {
              errors->back().reset();
              swap((*errors)[0], (*errors)[1]);
            } else if (errors->back()->step_time != current_step_time) {
              swap((*errors)[0], (*errors)[1]);
            }
          },
          box, db::get<history_tag>(*box), db::get<Tags::TimeStep>(*box),
          db::get<Tags::TimeStepper<TimeStepper>>(*box));
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
