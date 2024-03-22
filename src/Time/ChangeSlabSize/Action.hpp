// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/ChangeSlabSize/ChangeSlabSize.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
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
  using inbox_tags =
      tmpl::list<::Tags::ChangeSlabSize::NumberOfExpectedMessagesInbox,
                 ::Tags::ChangeSlabSize::NewSlabSizeInbox>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    if (not time_step_id.is_at_slab_boundary()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    auto& message_count_inbox =
        tuples::get<::Tags::ChangeSlabSize::NumberOfExpectedMessagesInbox>(
            inboxes);
    if (message_count_inbox.empty() or
        message_count_inbox.begin()->first != time_step_id.slab_number()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    auto& new_slab_size_inbox =
        tuples::get<::Tags::ChangeSlabSize::NewSlabSizeInbox>(inboxes);

    const auto slab_number = time_step_id.slab_number();
    const auto number_of_changes = [&slab_number](const auto& inbox) -> size_t {
      if (inbox.empty()) {
        return 0;
      }
      if (inbox.begin()->first == slab_number) {
        return inbox.begin()->second.size();
      }
      ASSERT(inbox.begin()->first >= slab_number,
             "Received data for a change at slab " << inbox.begin()->first
             << " but it is already slab " << slab_number);
      return 0;
    };

    const size_t expected_messages = number_of_changes(message_count_inbox);
    const size_t received_messages = number_of_changes(new_slab_size_inbox);
    ASSERT(expected_messages >= received_messages,
           "Received " << received_messages << " size change messages at slab "
                       << slab_number << ", but only expected "
                       << expected_messages);
    if (expected_messages != received_messages) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    message_count_inbox.erase(message_count_inbox.begin());

    const double new_slab_size =
        *alg::min_element(new_slab_size_inbox.begin()->second);
    new_slab_size_inbox.erase(new_slab_size_inbox.begin());

    const TimeStepper& time_stepper =
        db::get<::Tags::TimeStepper<TimeStepper>>(box);

    // Sometimes time steppers need to run with a fixed step size.
    // This is generally at the start of an evolution when the history
    // is in an unusual state.
    if (not time_stepper.can_change_step_size(
            time_step_id, db::get<::Tags::HistoryEvolvedVariables<>>(box))) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const auto current_slab = time_step_id.step_time().slab();

    const double new_slab_end =
        time_step_id.time_runs_forward()
            ? current_slab.start().value() + new_slab_size
            : current_slab.end().value() - new_slab_size;

    change_slab_size(make_not_null(&box), new_slab_end);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
