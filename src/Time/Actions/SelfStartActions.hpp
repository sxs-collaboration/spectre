// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Parallel/Actions/Goto.hpp"     // IWYU pragma: keep
#include "Parallel/Actions/TerminatePhase.hpp"     // IWYU pragma: keep
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/Actions/AdvanceTime.hpp"  // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

/// \ingroup TimeGroup
/// Definition of the integrator self-starting procedure.
///
/// The self-start procedure generates \f$N\f$ function values of
/// accuracy \f$O\left((\Delta t)^N\right)\f$, where \f$N\f$ is the
/// negative of the initial slab number.  To generate values, it
/// requires the time stepper to be a multistep integrator that will
/// produce an order-\f$k\f$-accurate result given \f$k-1\f$ history
/// values.
///
/// If the integrator is started from analytic history data or
/// requires no history (such as for a substep integrator), then the
/// initial slab number can be set to zero and no self-start steps
/// will be taken.
///
/// \details
/// To self-start a multistep integrator, the function is integrated
/// repeatedly with increasing accuracy.  A first order integrator
/// (Euler's method) requires no history values, so it can be used,
/// starting from the initial point, to generate a
/// first-order-accurate value at a later time.  We then reset to the
/// start conditions and use the new "history" value (at a discarded
/// point later in time) to take two steps with a second-order method.
/// These values are second-order-accurate despite the history only
/// being first-order because the calculation of the change in the
/// value multiplies the previous derivatives by a factor of
/// \f$\Delta t\f$.  The time and value are then again reset to their
/// starting values and we start again at third order, and so on.
///
/// The choice of performing the low-order integrations in the same
/// direction as the main integration makes this a _forward_
/// self-start procedure, as opposed to a _backward_ procedure that
/// produces values for times before the start time.  The primary
/// advantage of the forward version is that the solution is
/// guaranteed to exist after the start time, but not before.  It also
/// makes bookkeeping easier, as the reset after each order increase
/// is to the initial state, rather than to a time one step further
/// back for each order.  It does have the disadvantage, however, of
/// leaving a non-monotonic history at the end of the procedure, which
/// the main evolution loop must be able to handle.
///
/// Each time the state is reset the slab number is increased by one.
/// This ensures that the evaluations are considered to be ordered in
/// their evaluation order, even though they are not monotonic in
/// time.  When the slab number reaches zero the initialization
/// procedure is complete and history appropriate for use for an
/// integrator of order \f$N+1\f$ has been generated.
///
/// The self-start procedure performs all its evaluations before the
/// end of the first time step of the main evolution.  It is important
/// that none of the early steps fall at the same time as the
/// self-start history values, so the main evolution should not
/// decrease its step size on the first step after the procedure.
/// Additionally, the history times will not be monotonicly increasing
/// until \f$N\f$ steps have been taken.  The local-time-stepping
/// calculations require monotonic time, so local time-stepping should
/// not be initiated until the self-start values have expired from the
/// history.  These restrictions on step-size changing are checked in
/// the TimeStepper::can_change_step_size method.
namespace SelfStart {
/// Self-start tags
namespace Tags {
/// \ingroup TimeGroup
/// The initial value of a quantity.  The contents are stored in a
/// tuple to avoid putting duplicate tensors into the DataBox.
template <typename Tag>
struct InitialValue : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = std::tuple<typename Tag::type>;
};
}  // namespace Tags

/// Self-start actions
namespace Actions {
namespace detail {
template <typename System, bool HasPrimitives>
struct vars_to_save_impl {
  using type = tmpl::list<typename System::variables_tag>;
};

template <typename System>
struct vars_to_save_impl<System, true> {
  using type = tmpl::list<typename System::variables_tag,
                          typename System::primitive_variables_tag>;
};

template <typename System>
using vars_to_save = typename vars_to_save_impl<
    System, System::has_primitive_and_conservative_vars>::type;
}  // namespace detail

/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Prepares the evolution for time-stepper self-starting.
///
/// Stores the initial values of the variables and time step and sets
/// an appropriate step for self-starting.
///
/// \details The self-start procedure must take place within one slab,
/// and we want to avoid the end of the slab so that we don't have to
/// deal with another action advancing the slab on us.  There will be
/// problems if the main evolution tries to evaluate at a time that
/// the self-start procedure used before the self-start version falls
/// out of the history, so we have to make sure that does not happen.
/// We can't do that by making sure our steps are large enough to keep
/// ahead because of the slab restriction, so instead we have to make
/// the self-start step smaller to ensure no collisions.  The easiest
/// way to do that is to fit the entire procedure before the first
/// real step, so we pick an initialization time step of
/// \f$\Delta t/(N+1)\f$, for \f$\Delta t\f$ the initial evolution
/// time step and \f$N\f$ the number of history points to be
/// generated.
///
/// The original `Tags::TimeStep` and `Tags::Next<Tags::TimeStep>` are
/// temporarily stored separately during the self-start procedure, and restored
/// to their original values at the conclusion of the self-start procedure in
/// preparation for the main evolution, so that the initial time steps during
/// the evolution are appropriately set according to `Initialization` phase
/// values.
///
/// Uses:
/// - GlobalCache: nothing
/// - DataBox:
///   - Tags::TimeStepId
///   - Tags::TimeStep
///   - variables_tag
///   - primitive_variables_tag if the system has primitives
///
/// DataBox changes:
/// - Adds:
///   - SelfStart::Tags::InitialValue<Tags::TimeStep>
///   - SelfStart::Tags::InitialValue<variables_tag>
///   - SelfStart::Tags::InitialValue<primitive_variables_tag> if the system
///     has primitives
/// - Removes: nothing
/// - Modifies: Tags::TimeStep
template <typename System>
struct Initialize {
 private:
  template <typename TagsToSave>
  struct StoreInitialValues;

  template <typename... TagsToSave>
  struct StoreInitialValues<tmpl::list<TagsToSave...>> {
    using simple_tags = tmpl::list<Tags::InitialValue<TagsToSave>...>;

    template <typename Box>
    static void apply(Box& box) noexcept {
      ::Initialization::mutate_assign<simple_tags>(
          make_not_null(&box), std::make_tuple(db::get<TagsToSave>(box))...);
    }
  };

 public:
  using simple_tags = typename StoreInitialValues<
      tmpl::push_back<detail::vars_to_save<System>, ::Tags::TimeStep,
                      ::Tags::Next<::Tags::TimeStep>>>::simple_tags;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {

    const TimeDelta initial_step = db::get<::Tags::TimeStep>(box);
    // The slab number increments each time a new point is generated
    // until it reaches zero.
    const auto values_needed =
        -db::get<::Tags::Next<::Tags::TimeStepId>>(box).slab_number();
    TimeDelta self_start_step = initial_step;

    // Make sure the starting procedure fits in a slab.
    if (abs(self_start_step * values_needed).fraction() >= 1) {
      // Decrease the step so that the generated history will be
      // entirely before the first step.  This ensures we will not
      // generate any duplicate times in the history as we start the
      // real evolution.
      self_start_step /= (values_needed + 1);
    }

    StoreInitialValues<
        tmpl::push_back<detail::vars_to_save<System>, ::Tags::TimeStep,
                        ::Tags::Next<::Tags::TimeStep>>>::apply(box);
    db::mutate<::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>>(
        make_not_null(&box),
        [&self_start_step](
            const gsl::not_null<::TimeDelta*> time_step,
            const gsl::not_null<::TimeDelta*> next_time_step) noexcept {
          *time_step = self_start_step;
          *next_time_step = self_start_step;
        });

    return std::make_tuple(std::move(box));
  }
};

/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Resets the state for the next iteration if the current order is
/// complete, and exits the self-start loop if the required order has
/// been reached.
///
/// Uses:
/// - GlobalCache: nothing
/// - DataBox: Tags::Next<Tags::TimeStepId>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
template <typename ExitTag>
struct CheckForCompletion {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&, bool, size_t> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;

    const bool done_with_order =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box).is_at_slab_boundary();

    if (done_with_order) {
      tmpl::for_each<detail::vars_to_save<system>>([&box](auto tag) noexcept {
        using Tag = tmpl::type_from<decltype(tag)>;
        db::mutate<Tag>(
            make_not_null(&box),
            [](const gsl::not_null<typename Tag::type*> value,
               const std::tuple<typename Tag::type>& initial_value) noexcept {
              *value = get<0>(initial_value);
            },
            db::get<Tags::InitialValue<Tag>>(box));
      });
    }

    // The self start procedure begins with slab number
    // -number_of_past_steps and counts up.  When we reach 0 we should
    // start the evolution proper.  The first thing the evolution loop
    // will do is update the time, so here we need to check if the
    // next time should be the first real step.
    if (db::get<::Tags::Next<::Tags::TimeStepId>>(box).slab_number() == 0) {
      return {std::move(box), false,
              tmpl::index_of<ActionList, ::Actions::Label<ExitTag>>::value};
    }
    return {std::move(box), false,
            tmpl::index_of<ActionList, CheckForCompletion>::value + 1};
  }
};

/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// If we have taken enough steps for this order, set the next time to
/// the start time and increment the slab number
///
/// Uses:
/// - GlobalCache: nothing
/// - DataBox:
///   - Tags::HistoryEvolvedVariables
///   - Tags::SubstepTime
///   - Tags::TimeStepId
///   - Tags::TimeStep
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::Next<Tags::TimeStepId> if there is an order increase
struct CheckForOrderIncrease {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
    const auto& time = db::get<::Tags::SubstepTime>(box);
    const auto& time_step = db::get<::Tags::TimeStep>(box);
    const auto& history = db::get<::Tags::HistoryEvolvedVariables<>>(box);

    const Time required_time =
        (time_step.is_positive() ? time.slab().start() : time.slab().end()) +
        history.integration_order() * time_step;
    const bool done_with_order = time == required_time;

    if (done_with_order) {
      db::mutate<::Tags::Next<::Tags::TimeStepId>,
                 ::Tags::HistoryEvolvedVariables<>>(
          make_not_null(&box),
          [](const gsl::not_null<::TimeStepId*> next_time_id,
             const gsl::not_null<std::decay_t<decltype(history)>*>
                 mutable_history,
             const ::TimeStepId& current_time_id) noexcept {
            const Slab slab = current_time_id.step_time().slab();
            *next_time_id =
                TimeStepId(current_time_id.time_runs_forward(),
                           current_time_id.slab_number() + 1,
                           current_time_id.time_runs_forward() ? slab.start()
                                                               : slab.end());
            mutable_history->integration_order(
                mutable_history->integration_order() + 1);
          },
          db::get<::Tags::TimeStepId>(box));
    }

    return {std::move(box)};
  }
};

/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Cleans up after the self-start procedure
///
/// Resets the time step to that requested for the evolution and
/// removes temporary self-start data.
///
/// Uses:
/// - GlobalCache: nothing
/// - DataBox: SelfStart::Tags::InitialValue<Tags::TimeStep>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes:
///   - All SelfStart::Tags::InitialValue tags
/// - Modifies: Tags::TimeStep
struct Cleanup {
 private:
  template <typename T>
  struct is_a_initial_value : tt::is_a<Tags::InitialValue, T> {};

  using initial_step_tag = Tags::InitialValue<::Tags::TimeStep>;

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Reset the time step to the value requested by the user.  The
    // variables were reset in CheckForCompletion.
    db::mutate<::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>>(
        make_not_null(&box),
        [](const gsl::not_null<::TimeDelta*> time_step,
           const gsl::not_null<::TimeDelta*> next_time_step,
           const std::tuple<::TimeDelta>& initial_step,
           const std::tuple<::TimeDelta>& initial_next_step) noexcept {
          *time_step = get<0>(initial_step);
          *next_time_step = get<0>(initial_next_step);
        },
        db::get<initial_step_tag>(box),
        db::get<Tags::InitialValue<::Tags::Next<::Tags::TimeStep>>>(box));
    using remove_tags = tmpl::filter<DbTags, is_a_initial_value<tmpl::_1>>;
    // reset each tag to default constructed values to reduce memory usage (Data
    // structures like `DataVector`s and `Tensor`s have negligible memory usage
    // when default-constructed). This tends to be better for build time than
    // constructing more DataBox types with and without this handful of tags.
    tmpl::for_each<remove_tags>(
        [&box](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          db::mutate<tag>(make_not_null(&box), [](auto tag_value) noexcept {
            *tag_value = std::decay_t<decltype(*tag_value)>{};
          });
        });
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions

namespace detail {
struct PhaseStart;
struct PhaseEnd;
}  // namespace detail

/// \ingroup TimeGroup
/// The list of actions required to self-start an integrator.
///
/// \tparam StepActions List of actions computing and recording the
/// system derivative and updating the evolved variables (but not the
/// time).
///
/// \see SelfStart
template <typename StepActions, typename System>
using self_start_procedure = tmpl::flatten<tmpl::list<
// clang-format off
    SelfStart::Actions::Initialize<System>,
    ::Actions::Label<detail::PhaseStart>,
    SelfStart::Actions::CheckForCompletion<detail::PhaseEnd>,
    ::Actions::AdvanceTime,
    SelfStart::Actions::CheckForOrderIncrease,
    StepActions,
    ::Actions::Goto<detail::PhaseStart>,
    ::Actions::Label<detail::PhaseEnd>,
    SelfStart::Actions::Cleanup,
    ::Actions::AdvanceTime,
    Parallel::Actions::TerminatePhase>>;
// clang-format on
}  // namespace SelfStart
