// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <pup.h>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <typename StepChooserRegistrars>
class StepChooser;
namespace Tags {
struct DataBox;
template <typename Tag>
struct Next;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace ChangeSlabSize_detail {
struct NewSlabSizeInbox
    : public Parallel::InboxInserters::MemberInsert<NewSlabSizeInbox> {
  using temporal_id = int64_t;
  using type = std::map<temporal_id, std::unordered_multiset<double>>;
};

// This inbox doesn't receive any data, it just counts messages (using
// the size of the multiset).  Whenever a message is sent to the
// NewSlabSizeInbox, another message is sent here synchronously, so
// the count here is the number of messages we expect in the
// NewSlabSizeInbox.
struct NumberOfExpectedMessagesInbox
    : public Parallel::InboxInserters::MemberInsert<
          NumberOfExpectedMessagesInbox> {
  using temporal_id = int64_t;
  using NoData = std::tuple<>;
  using type = std::map<temporal_id,
                        std::unordered_multiset<NoData, boost::hash<NoData>>>;
};

struct StoreNewSlabSize {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const int64_t slab_number,
                    const double slab_size) noexcept {
    Parallel::receive_data<ChangeSlabSize_detail::NewSlabSizeInbox>(
        *Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
             .ckLocal(),
        slab_number, slab_size);
  }
};
}  // namespace ChangeSlabSize_detail

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Adjust the slab size based on previous executions of
/// Events::ChangeSlabSize
///
/// Uses:
/// - GlobalCache:
///   - Tags::TimeStepperBase
/// - DataBox:
///   - Tags::HistoryEvolvedVariables
///   - Tags::TimeStep
///   - Tags::TimeStepId
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
      tmpl::list<ChangeSlabSize_detail::NumberOfExpectedMessagesInbox,
                 ChangeSlabSize_detail::NewSlabSizeInbox>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& time_step_id = db::get<Tags::TimeStepId>(box);

    if (not time_step_id.is_at_slab_boundary()) {
      return std::forward_as_tuple(std::move(box));
    }

    auto& message_count_inbox =
        tuples::get<ChangeSlabSize_detail::NumberOfExpectedMessagesInbox>(
            inboxes);
    if (message_count_inbox.empty() or
        message_count_inbox.begin()->first != time_step_id.slab_number()) {
      return std::forward_as_tuple(std::move(box));
    }
    message_count_inbox.erase(message_count_inbox.begin());

    auto& new_slab_size_inbox =
        tuples::get<ChangeSlabSize_detail::NewSlabSizeInbox>(inboxes);
    const double new_slab_size =
        *alg::min_element(new_slab_size_inbox.begin()->second);
    new_slab_size_inbox.erase(new_slab_size_inbox.begin());

    const TimeStepper& time_stepper = db::get<Tags::TimeStepper<>>(box);

    // Sometimes time steppers need to run with a fixed step size.
    // This is generally at the start of an evolution when the history
    // is in an unusual state.
    if (not time_stepper.can_change_step_size(
            time_step_id, db::get<Tags::HistoryEvolvedVariables<>>(box))) {
      return std::forward_as_tuple(std::move(box));
    }

    const auto& current_step = db::get<Tags::TimeStep>(box);
    const auto& current_slab = current_step.slab();

    const auto new_slab =
        current_step.is_positive()
            ? current_slab.with_duration_from_start(new_slab_size)
            : current_slab.with_duration_to_end(new_slab_size);

    const auto new_step = current_step.with_slab(new_slab);

    // We are at a slab boundary, so the substep is 0.
    const auto new_time_step_id =
        TimeStepId(time_step_id.time_runs_forward(), time_step_id.slab_number(),
                   time_step_id.step_time().with_slab(new_slab));

    const auto new_next_time_step_id =
        time_stepper.next_time_id(new_time_step_id, new_step);

    db::mutate<Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
               Tags::Next<Tags::TimeStep>, Tags::TimeStepId>(
        make_not_null(&box),
        [&new_next_time_step_id, &new_step, &new_time_step_id](
            const gsl::not_null<TimeStepId*> next_time_step_id,
            const gsl::not_null<TimeDelta*> time_step,
            const gsl::not_null<TimeDelta*> next_time_step,
            const gsl::not_null<TimeStepId*> local_time_step_id) noexcept {
          *next_time_step_id = new_next_time_step_id;
          *time_step = new_step;
          *next_time_step = new_step;
          *local_time_step_id = new_time_step_id;
        });

    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    const auto& time_step_id = db::get<Tags::TimeStepId>(box);
    if (not time_step_id.is_at_slab_boundary()) {
      return true;
    }

    const auto& message_count_inbox =
        tuples::get<ChangeSlabSize_detail::NumberOfExpectedMessagesInbox>(
            inboxes);
    const auto& new_slab_size_inbox =
        tuples::get<ChangeSlabSize_detail::NewSlabSizeInbox>(inboxes);

    const auto slab_number = time_step_id.slab_number();
    const auto number_of_changes = [&slab_number](
                                       const auto& inbox) noexcept->size_t {
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
    return expected_messages == received_messages;
  }
};
}  // namespace Actions

namespace Events {
template <typename StepChooserRegistrars, typename EventRegistrars>
class ChangeSlabSize;

namespace Registrars {
template <typename StepChooserRegistrars>
using ChangeSlabSize =
    Registration::Registrar<Events::ChangeSlabSize, StepChooserRegistrars>;
}  // namespace Registrars

/// \ingroup TimeGroup
/// %Trigger a slab size change.
///
/// The new size will be the minimum suggested by any of the provided
/// step choosers on any element.  This requires a global reduction,
/// so it is possible to delay the change until a later slab to avoid
/// a global synchronization.  The actual change is carried out by
/// Actions::ChangeSlabSize.
///
/// When running with global time-stepping, the slab size and step
/// size are the same, so this adjusts the step size used by the time
/// integration.  With local time-stepping this controls the interval
/// between times when the sequences of steps on all elements are
/// forced to align.
template <typename StepChooserRegistrars,
          typename EventRegistrars =
              tmpl::list<Registrars::ChangeSlabSize<StepChooserRegistrars>>>
class ChangeSlabSize : public Event<EventRegistrars> {
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<int64_t, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<double, funcl::Min<>>>;

 public:
  /// \cond
  explicit ChangeSlabSize(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ChangeSlabSize);  // NOLINT
  /// \endcond

  struct StepChoosers {
    static constexpr Options::String help = "Limits on slab size";
    using type =
        std::vector<std::unique_ptr<StepChooser<StepChooserRegistrars>>>;
    static size_t lower_bound_on_size() noexcept { return 1; }
  };

  struct DelayChange {
    static constexpr Options::String help = "Slabs to wait before changing";
    using type = uint64_t;
  };

  using options = tmpl::list<StepChoosers, DelayChange>;
  static constexpr Options::String help =
      "Trigger a slab size change chosen by the provided step choosers.\n"
      "The actual changing of the slab size can be delayed until a later\n"
      "slab to improve parallelization.";

  ChangeSlabSize() = default;
  ChangeSlabSize(
      std::vector<std::unique_ptr<StepChooser<StepChooserRegistrars>>>
          step_choosers,
      const uint64_t delay_change) noexcept
      : step_choosers_(std::move(step_choosers)), delay_change_(delay_change) {}

  using argument_tags = tmpl::list<Tags::TimeStepId, Tags::DataBox>;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const TimeStepId& time_step_id,
                  const db::DataBox<DbTags>& box_for_step_choosers,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/) const noexcept {
    const auto next_changable_slab = time_step_id.is_at_slab_boundary()
                                         ? time_step_id.slab_number()
                                         : time_step_id.slab_number() + 1;
    const auto slab_to_change =
        next_changable_slab + static_cast<int64_t>(delay_change_);

    double desired_slab_size = std::numeric_limits<double>::infinity();
    for (const auto& step_chooser : step_choosers_) {
      desired_slab_size =
          std::min(desired_slab_size,
                   step_chooser->desired_slab(
                       time_step_id.step_time().slab().duration().value(),
                       box_for_step_choosers, cache));
    }

    const auto& component_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const auto& self_proxy = component_proxy[array_index];
    // This message is sent synchronously, so it is guaranteed to
    // arrive before the ChangeSlabSize action is called.
    Parallel::receive_data<
        ChangeSlabSize_detail::NumberOfExpectedMessagesInbox>(
        *self_proxy.ckLocal(), slab_to_change,
        ChangeSlabSize_detail::NumberOfExpectedMessagesInbox::NoData{});
    Parallel::contribute_to_reduction<ChangeSlabSize_detail::StoreNewSlabSize>(
        ReductionData(slab_to_change, desired_slab_size), self_proxy,
        component_proxy);
  }

  bool needs_evolved_variables() const noexcept override {
    // This depends on the chosen StepChoosers, but they don't have a
    // way to report this information so we just return true to be
    // safe.
    return true;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    Event<EventRegistrars>::pup(p);
    p | step_choosers_;
    p | delay_change_;
  }

 private:
  std::vector<std::unique_ptr<StepChooser<StepChooserRegistrars>>>
      step_choosers_;
  uint64_t delay_change_ = std::numeric_limits<uint64_t>::max();
};

/// \cond
template <typename StepChooserRegistrars, typename EventRegistrars>
PUP::able::PUP_ID
    ChangeSlabSize<StepChooserRegistrars, EventRegistrars>::my_PUP_ID =
        0;  // NOLINT
/// \endcond
}  // namespace Events
