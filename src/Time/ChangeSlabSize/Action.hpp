// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/ChangeSlabSize/ChangeSlabSize.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Adjust the slab size based on previous executions of
/// Events::ChangeSlabSize
///
/// Uses:
/// - DataBox:
///   - Tags::HistoryEvolvedVariables
///   - Tags::TimeStep
///   - Tags::TimeStepId
///   - Tags::TimeStepper<TimeStepper>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::Next<Tags::TimeStepId>
///   - Tags::TimeStep
///   - Tags::TimeStepId
struct ChangeSlabSize {
  using simple_tags =
      tmpl::list<::Tags::ChangeSlabSize::NewSlabSize,
                 ::Tags::ChangeSlabSize::NumberOfExpectedMessages>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    if (not time_step_id.is_at_slab_boundary()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    TimeStepRequestProcessor step_requests(time_step_id.time_runs_forward());

    const auto slab_number = time_step_id.slab_number();
    const auto& expected_messages_map =
        db::get<::Tags::ChangeSlabSize::NumberOfExpectedMessages>(box);
    if (not expected_messages_map.empty() and
        expected_messages_map.begin()->first == slab_number) {
      const size_t expected_messages = expected_messages_map.begin()->second;
      ASSERT(expected_messages > 0,
             "Should only create map entries when sending messages.");

      const auto& slab_size_messages =
          db::get<::Tags::ChangeSlabSize::NewSlabSize>(box);
      if (slab_size_messages.empty()) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
      const int64_t first_received_change_slab =
          slab_size_messages.begin()->first;

      ASSERT(first_received_change_slab >= slab_number,
             "Received data for a change at slab " << first_received_change_slab
             << " but it is already slab " << slab_number);
      if (first_received_change_slab != slab_number) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }

      const auto& received_changes = slab_size_messages.begin()->second;
      ASSERT(expected_messages >= received_changes.size(),
             "Received " << received_changes.size()
                         << " size change messages at slab " << slab_number
                         << ", but only expected " << expected_messages);
      if (received_changes.size() != expected_messages) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }

      // We have all the data we need.

      step_requests =
          alg::accumulate(slab_size_messages.begin()->second, step_requests);

      db::mutate<::Tags::ChangeSlabSize::NumberOfExpectedMessages,
                 ::Tags::ChangeSlabSize::NewSlabSize>(
          [](const gsl::not_null<std::map<int64_t, size_t>*> expected,
             const gsl::not_null<
                 std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                 sizes) {
            expected->erase(expected->begin());
            sizes->erase(sizes->begin());
          },
          make_not_null(&box));
    }
    const double new_slab_end = step_requests.step_end(
        time_step_id.step_time().value(),
        db::get<::Tags::ChangeSlabSize::SlabSizeGoal>(box));

    if (const auto new_goal = step_requests.new_step_size_goal();
        new_goal.has_value()) {
      db::mutate<::Tags::ChangeSlabSize::SlabSizeGoal>(
          [&](const gsl::not_null<double*> slab_size_goal) {
            *slab_size_goal = *new_goal;
          },
          make_not_null(&box));
    }

    const TimeStepper& time_stepper =
        db::get<::Tags::TimeStepper<TimeStepper>>(box);

    // Sometimes time steppers need to run with a fixed step size.
    // This is generally at the start of an evolution when the history
    // is in an unusual state.
    if (time_stepper.can_change_step_size(
            time_step_id, db::get<::Tags::HistoryEvolvedVariables<>>(box))) {
      change_slab_size(make_not_null(&box), new_slab_end);
    }

    step_requests.error_on_hard_limit(
        db::get<::Tags::TimeStep>(box).value(),
        (time_step_id.step_time() + db::get<::Tags::TimeStep>(box)).value());
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
