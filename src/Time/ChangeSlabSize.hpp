// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct AdaptiveSteppingDiagnostics;
template <typename Tag>
struct Next;
struct TimeStep;
struct TimeStepId;
}  // namespace Tags
/// \endcond

/// \ingroup TimeGroup
/// Change the slab size, updating all quantities in the DataBox
/// depending on it.
///
/// The end of the slab (in the appropriate direction for the
/// evolution's flow of time) will be set to \p new_slab_end.  The
/// time step is set to the same fraction of the new slab as it was of
/// the old.
template <typename DbTags>
void change_slab_size(const gsl::not_null<db::DataBox<DbTags>*> box,
                      const double new_slab_end) {
  const TimeStepId& old_time_step_id = db::get<Tags::TimeStepId>(*box);

  ASSERT(old_time_step_id.is_at_slab_boundary(),
         "Cannot change slab size in middle of slab: " << old_time_step_id);

  const Slab old_slab = old_time_step_id.step_time().slab();
  const double old_slab_end = old_time_step_id.time_runs_forward()
                                  ? old_slab.end().value()
                                  : old_slab.start().value();

  if (new_slab_end == old_slab_end) {
    return;
  }

  const TimeDelta& old_time_step = db::get<Tags::TimeStep>(*box);

  const Slab new_slab = old_time_step_id.time_runs_forward()
                            ? Slab(old_slab.start().value(), new_slab_end)
                            : Slab(new_slab_end, old_slab.end().value());
  // We are at a slab boundary, so the substep is 0.
  const TimeStepId new_time_step_id(
      old_time_step_id.time_runs_forward(), old_time_step_id.slab_number(),
      old_time_step_id.step_time().with_slab(new_slab));
  const auto new_time_step = old_time_step.with_slab(new_slab);

  const auto new_next_time_step_id =
      db::get<Tags::TimeStepper<>>(*box).next_time_id(new_time_step_id,
                                                      new_time_step);

  db::mutate_apply<
      tmpl::push_front<Tags::get_all_history_tags<DbTags>,
                       ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
                       ::Tags::Next<::Tags::TimeStep>, ::Tags::TimeStepId,
                       ::Tags::AdaptiveSteppingDiagnostics>,
      tmpl::list<>>(
      [&new_next_time_step_id, &new_time_step, &new_time_step_id](
          const gsl::not_null<TimeStepId*> next_time_step_id,
          const gsl::not_null<TimeDelta*> time_step,
          const gsl::not_null<TimeDelta*> next_time_step,
          const gsl::not_null<TimeStepId*> local_time_step_id,
          const gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
          const auto... histories) {
        const auto update_history = [&](const auto history) {
          if (not history->empty() and
              history->back().time_step_id == *local_time_step_id) {
            ASSERT(history->at_step_start(),
                   "Cannot change step size with substep data.");
            history->back().time_step_id = new_time_step_id;
          }
          return 0;
        };
        expand_pack(update_history(histories)...);

        *next_time_step_id = new_next_time_step_id;
        *time_step = new_time_step;
        *next_time_step = new_time_step;
        *local_time_step_id = new_time_step_id;
        ++diags->number_of_slab_size_changes;
      },
      box);
}
