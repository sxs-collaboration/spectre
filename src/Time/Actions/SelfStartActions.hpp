// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/GotoAction.hpp"  // IWYU pragma: keep
#include "Time/Actions/AdvanceTime.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep // for item_type<Tags::TimeStep>
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
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
  static std::string name() noexcept {
    return "InitialValue(" + Tag::name() + ")";
  }
  using tag = Tag;
  using type = std::tuple<db::item_type<Tag>>;
};
}  // namespace Tags

/// Self-start actions
namespace Actions {
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
/// Uses:
/// - ConstGlobalCache: nothing
/// - DataBox:
///   - Tags::TimeId
///   - Tags::TimeStep
///   - variables_tag
///
/// DataBox changes:
/// - Adds:
///   - SelfStart::Tags::InitialValue<Tags::TimeStep>
///   - SelfStart::Tags::InitialValue<variables_tag>
/// - Removes: nothing
/// - Modifies: Tags::TimeStep
struct Initialize {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using variables_tag = typename Metavariables::system::variables_tag;

    const TimeDelta initial_step = db::get<::Tags::TimeStep>(box);
    // The slab number increments each time a new point is generated
    // until it reaches zero.
    const auto values_needed =
        -db::get<::Tags::Next<::Tags::TimeId>>(box).slab_number();
    const TimeDelta self_start_step = initial_step / (values_needed + 1);

    db::mutate<::Tags::TimeStep>(
        make_not_null(&box), [&self_start_step](
                                 const gsl::not_null<
                                     db::item_type<::Tags::TimeStep>*>
                                     time_step) noexcept {
          *time_step = self_start_step;
        });

    return std::make_tuple(
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<Tags::InitialValue<::Tags::TimeStep>,
                                          Tags::InitialValue<variables_tag>>>(
            std::move(box), std::make_tuple(initial_step),
            std::make_tuple(db::get<variables_tag>(box))));
  }
};

/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Terminates the self-start phase if the required order has been
/// reached.
///
/// Uses:
/// - ConstGlobalCache: nothing
/// - DataBox: Tags::Next<Tags::TimeId>
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
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // The self start procedure begins with slab number
    // -number_of_past_steps and counts up.  When we reach 0 we should
    // start the evolution proper.  The first thing the evolution loop
    // will do is update the time, so here we need to check if the
    // next time should be the first real step.
    return std::tuple<db::DataBox<DbTags>&&, bool, size_t>(
        std::move(box), false,
        db::get<::Tags::Next<::Tags::TimeId>>(box).slab_number() == 0
            ? tmpl::index_of<ActionList, ::Actions::Label<ExitTag>>::value
            : tmpl::index_of<ActionList, CheckForCompletion>::value + 1);
    // Once we have full support for phases this action should
    // terminate the phase:
    // return std::tuple<db::DataBox<DbTags>&&, bool>(
    //     std::move(box),
    //     db::get<::Tags::Next<::Tags::TimeId>>(box).slab_number() == 0);
  }
};

/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// If we have taken enough steps for this order, set the next time to
/// the start time and increment the slab number
///
/// Uses:
/// - ConstGlobalCache: nothing
/// - DataBox:
///   - Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>
///   - Tags::Time
///   - Tags::TimeId
///   - Tags::TimeStep
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::Next<Tags::TimeId> if there is an order increase
struct CheckForOrderIncrease {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using variables_tag = typename Metavariables::system::variables_tag;

    const auto& time = db::get<::Tags::Time>(box);
    const auto& time_step = db::get<::Tags::TimeStep>(box);
    const auto& history = db::get<::Tags::HistoryEvolvedVariables<
        variables_tag, db::add_tag_prefix<::Tags::dt, variables_tag>>>(box);

    const Time required_time =
        (time_step.is_positive() ? time.slab().start() : time.slab().end()) +
        (history.size() + 1) * time_step;
    const bool done_with_order = time == required_time;

    if (done_with_order) {
      db::mutate<::Tags::Next<::Tags::TimeId>>(
          make_not_null(&box),
          [](const gsl::not_null<db::item_type<::Tags::Next<::Tags::TimeId>>*>
                 next_time_id,
             const db::item_type<::Tags::TimeId>& current_time_id) noexcept {
            const Slab slab = current_time_id.time().slab();
            *next_time_id =
                TimeId(current_time_id.time_runs_forward(),
                       current_time_id.slab_number() + 1,
                       current_time_id.time_runs_forward() ? slab.start()
                                                           : slab.end());
          },
          db::get<::Tags::TimeId>(box));
    }

    return std::forward_as_tuple(std::move(box));
  }
};

/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Jumps to the start of the self-start algorithm (skipping taking a
/// step from the last point) if the generation of the points for the
/// current order is complete.
///
/// Uses:
/// - ConstGlobalCache: nothing
/// - DataBox:
///   - Tags::Next<Tags::TimeId>
///   - SelfStart::Tags::InitialValue<variables_tag>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: variables_tag if there is an order increase
template <typename RestartTag>
struct StartNextOrderIfReady {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using variables_tag = typename Metavariables::system::variables_tag;

    constexpr size_t restart_index =
        tmpl::index_of<ActionList, ::Actions::Label<RestartTag>>::value + 1;
    constexpr size_t continue_index =
        tmpl::index_of<ActionList, StartNextOrderIfReady>::value + 1;

    const bool done_with_order =
        db::get<::Tags::Next<::Tags::TimeId>>(box).is_at_slab_boundary();

    if (done_with_order) {
      db::mutate<variables_tag>(
          make_not_null(&box),
          [](const gsl::not_null<db::item_type<variables_tag>*> variables,
             const db::item_type<Tags::InitialValue<variables_tag>>&
                 initial_variables) noexcept {
            *variables = get<0>(initial_variables);
          },
          db::get<Tags::InitialValue<variables_tag>>(box));
    }

    return std::tuple<db::DataBox<DbTags>&&, bool, size_t>(
        std::move(box), false,
        done_with_order ? restart_index : continue_index);
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
/// - ConstGlobalCache: nothing
/// - DataBox: SelfStart::Tags::InitialValue<Tags::TimeStep>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes:
///   - SelfStart::Tags::InitialValue<Tags::TimeStep>
///   - SelfStart::Tags::InitialValue<variables_tag>
/// - Modifies: Tags::TimeStep
struct Cleanup {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using variables_tag = typename Metavariables::system::variables_tag;
    using initial_step_tag = Tags::InitialValue<::Tags::TimeStep>;

    // Reset the time step to the value requested by the user.  The
    // variables were reset in StartNextOrderIfReady.
    db::mutate<::Tags::TimeStep>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<::Tags::TimeStep>*> time_step,
           const db::item_type<initial_step_tag>& initial_step) noexcept {
          *time_step = get<0>(initial_step);
        },
        db::get<initial_step_tag>(box));

    return std::make_tuple(
        db::create_from<db::RemoveTags<initial_step_tag,
                                       Tags::InitialValue<variables_tag>>>(
            std::move(box)));
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
/// \tparam ComputeRhs Action or list of actions computing and
/// recording the system derivative.
/// \tparam UpdateVariables Action or list of actions updating the
/// evolved variables (but not the time).
///
/// \see SelfStart
// clang-format off
template <typename ComputeRhs, typename UpdateVariables>
using self_start_procedure = tmpl::flatten<tmpl::list<
    SelfStart::Actions::Initialize,
    ::Actions::Label<detail::PhaseStart>,
    SelfStart::Actions::CheckForCompletion<detail::PhaseEnd>,
    ::Actions::AdvanceTime,
    SelfStart::Actions::CheckForOrderIncrease,
    ComputeRhs,
    SelfStart::Actions::StartNextOrderIfReady<detail::PhaseStart>,
    UpdateVariables,
    ::Actions::Goto<detail::PhaseStart>,
    ::Actions::Label<detail::PhaseEnd>,
    SelfStart::Actions::Cleanup>>;
// clang-format on
}  // namespace SelfStart
