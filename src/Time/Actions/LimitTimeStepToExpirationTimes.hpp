// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Utilities.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FractionUtilities.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename Tag>
struct Next;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Make sure step size does not exceed FunctionOfTime expiration times
///
/// \details For time steppers with substeps, functions of time might expire
/// at a time earlier than one of the substeps; however, the functions of time
/// are both needed at each substep and updated only after taking a full step.
/// This can result in quiescence, i.e., in waiting to complete the next
/// substep until the functions of time are updated, but the functions of time
/// will only be updated after the next step is taken. To avoid this situation,
/// at the start of each full step, this action checks whether the current
/// step size exceeds the time remaining before the first function of time
/// expires. If it does, then the step size is adjusted to be earlier than
/// the expiration time of the next function of time to expire. Note that
/// if the time stepper does not use substeps, this action does nothing.
///
/// Uses:
/// - DataBox:
///   - Tags::Time
///   - Tags::TimeStep
///   - Tags::TimeStepId
///   - Tags::TimeStepper<>
///   - domain::Tags::FunctionsOfTime
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::TimeStep
struct LimitTimeStepToExpirationTimes {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    // First, check whether the time stepper uses substeps
    const TimeStepper& time_stepper = db::get<Tags::TimeStepper<>>(box);
    if (time_stepper.number_of_substeps() > 0) {
      // Next, check if the current time step is at the beginning of a full step
      const auto& time_step_id = db::get<Tags::TimeStepId>(box);
      if (time_step_id.substep() == 0) {
        // Get the functions of time and, if not empty, loop over them,
        // finding the minimum expiration time
        const auto& functions_of_time =
            get<domain::Tags::FunctionsOfTime>(cache);
        if (not functions_of_time.empty()) {
          double min_expiration_time{std::numeric_limits<double>::max()};
          for (const auto& [name, f_of_t] : functions_of_time) {
            if (f_of_t->time_bounds()[1] < min_expiration_time) {
              min_expiration_time = f_of_t->time_bounds()[1];
            }
          }

          ASSERT(db::get<Tags::Time>(box) <= min_expiration_time,
                 "Time (" << std::setprecision(20) << db::get<Tags::Time>(box)
                          << ") is greater than expiration time ("
                          << min_expiration_time << ") by an amount of "
                          << db::get<Tags::Time>(box) - min_expiration_time);
          if (db::get<Tags::Time>(box) == min_expiration_time) {
            return {Parallel::AlgorithmExecution::Continue, std::nullopt};
          }

          const auto& initial_time_step{db::get<Tags::TimeStep>(box)};
          const double step_end =
              (time_step_id.step_time() + initial_time_step).value();
          if (time_step_id.step_time().value() +
                  2.0 * initial_time_step.value() <=
              min_expiration_time) {
            // The expiration times are far in the future (more than
            // two steps).  No need to adjust anything.
            return {Parallel::AlgorithmExecution::Continue, std::nullopt};
          }

          const double start{initial_time_step.slab().start().value()};
          Slab new_slab{};
          if (step_end + slab_rounding_error(time_step_id.step_time()) >=
              min_expiration_time) {
            // The minimum expiration time is within the next step (or
            // within roundoff).  Step to it.
            new_slab = Slab(start, min_expiration_time);
          } else {
            // The minimum expiration time is not within the next
            // step, but is within two steps.  Shrink the step size to
            // avoid taking a tiny step after this one.

            // new_slab = initial_time_step.slab().with_duration_to_end(
            //     0.51 * (min_expiration_time - start));
            // Note: 0.51 instead of 0.5 to guarantee no roundoff affecting
            // what happens on the next step.
            new_slab = Slab(start, 0.51 * min_expiration_time + 0.49 * start);
          }

          ASSERT(initial_time_step.fraction() == 1,
                 "This action does not currently support local time stepping, "
                 "but the time step fraction is "
                     << initial_time_step.fraction() << " instead of 1");
          const TimeDelta new_time_step = new_slab.duration();
          const TimeStepId new_time_step_id{
              time_step_id.time_runs_forward(), time_step_id.slab_number(),
              time_step_id.step_time().with_slab(new_slab)};
          const TimeStepId new_next_time_step_id =
              time_stepper.next_time_id(new_time_step_id, new_time_step);
          const TimeDelta new_next_time_step{
              new_next_time_step_id.step_time().slab(),
              initial_time_step.fraction()};

          db::mutate<::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                     ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>>(
              make_not_null(&box),
              [&new_time_step, &new_next_time_step, &new_time_step_id,
               &new_next_time_step_id](
                  const gsl::not_null<TimeDelta*> time_step,
                  const gsl::not_null<TimeDelta*> next_time_step,
                  const gsl::not_null<TimeStepId*> time_step_id,
                  const gsl::not_null<TimeStepId*> next_time_step_id) {
                *time_step = new_time_step;
                *next_time_step = new_next_time_step;
                *time_step_id = new_time_step_id;
                *next_time_step_id = new_next_time_step_id;
              });
        }
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
