// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class TimeDelta;
class TimeId;
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Adjust the step size for local time stepping
///
/// Uses:
/// - ConstGlobalCache:
///   - OptionTags::StepChoosers<StepChooserRegistrars>
///   - OptionTags::StepController
///   - OptionTags::TimeStepper
/// - DataBox:
///   - Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>
///   - Tags::TimeId
///   - Tags::TimeStep
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::Next<Tags::TimeId>, Tags::TimeStep
template <typename StepChooserRegistrars>
struct ChangeStepSize {
  using step_choosers_tag = OptionTags::StepChoosers<StepChooserRegistrars>;
  using const_global_cache_tags =
      tmpl::list<step_choosers_tag, OptionTags::StepController>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    static_assert(Metavariables::local_time_stepping,
                  "ChangeStepSize can only be used with local time-stepping.");

    using variables_tag = typename Metavariables::system::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;

    const LtsTimeStepper& time_stepper =
        Parallel::get<OptionTags::TimeStepper>(cache);
    const auto& step_choosers = Parallel::get<step_choosers_tag>(cache);
    const auto& step_controller =
        Parallel::get<OptionTags::StepController>(cache);

    const auto& time_id = db::get<Tags::TimeId>(box);

    if (not time_stepper.can_change_step_size(
            time_id,
            db::get<
                Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>>(
                box))) {
      return std::forward_as_tuple(std::move(box));
    }

    const auto& current_step = db::get<Tags::TimeStep>(box);

    // The step choosers return the magnitude of the desired step, so
    // we always want the minimum requirement, but we have to negate
    // the final answer if time is running backwards.
    double desired_step = std::numeric_limits<double>::infinity();
    for (const auto& step_chooser : step_choosers) {
      desired_step =
          std::min(desired_step, step_chooser->desired_step(box, cache));
    }
    if (not current_step.is_positive()) {
      desired_step = -desired_step;
    }

    const auto new_step =
        step_controller.choose_step(time_id.time(), desired_step);
    if (new_step != current_step) {
      const auto new_next_time_id =
          time_stepper.next_time_id(time_id, new_step);

      db::mutate<Tags::Next<Tags::TimeId>, Tags::TimeStep>(
          make_not_null(&box),
          [&new_next_time_id, &new_step](
              const gsl::not_null<TimeId*> next_time_id,
              const gsl::not_null<TimeDelta*> time_step) noexcept {
            *next_time_id = new_next_time_id;
            *time_step = new_step;
          });
    }

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
