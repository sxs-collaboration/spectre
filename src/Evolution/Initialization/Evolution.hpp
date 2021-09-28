// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Evolution_detail {
// Global time stepping
template <typename Metavariables, typename DbTagsList,
          Requires<not Metavariables::local_time_stepping> = nullptr>
TimeDelta get_initial_time_step(const Time& initial_time,
                                const double initial_dt_value,
                                const db::DataBox<DbTagsList>& /*box*/) {
  return (initial_dt_value > 0.0 ? 1 : -1) * initial_time.slab().duration();
}

// Local time stepping
template <typename Metavariables, typename DbTagsList,
          Requires<Metavariables::local_time_stepping> = nullptr>
TimeDelta get_initial_time_step(const Time& initial_time,
                                const double initial_dt_value,
                                const db::DataBox<DbTagsList>& box) {
  const auto& step_controller = db::get<Tags::StepController>(box);
  return step_controller.choose_step(initial_time, initial_dt_value);
}
}  // namespace Evolution_detail

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Initialize items related to time, such as the time step
///
/// Since we have not started the evolution yet, we initialize the state
/// _before_ the initial time. So `Tags::TimeStepId` is undefined at this point,
/// and `Tags::Next<Tags::TimeStepId>` is the initial time.
///
/// DataBox changes:
/// - Adds:
///   * Tags::TimeStepId
///   * `Tags::Next<Tags::TimeStepId>`
///   * Tags::TimeStep
///   * Tags::Time
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note HistoryEvolvedVariables is allocated, but needs to be initialized
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <typename Metavariables>
struct TimeAndTimeStep {
  using initialization_tags = tmpl::flatten<
      tmpl::list<Tags::InitialTime, Tags::InitialTimeDelta,
                 Tags::InitialSlabSize<Metavariables::local_time_stepping>,
                 tmpl::conditional_t<
                     Metavariables::local_time_stepping,
                     tmpl::list<::Tags::IsUsingTimeSteppingErrorControl<>,
                                ::Tags::TimeStepper<LtsTimeStepper>,
                                ::Tags::StepChoosers, ::Tags::StepController>,
                     tmpl::list<::Tags::NeverUsingTimeSteppingErrorControl,
                                ::Tags::TimeStepper<TimeStepper>>>>>;

  using initialization_tags_to_keep =
      tmpl::flatten<tmpl::list<tmpl::conditional_t<
          Metavariables::local_time_stepping,
          tmpl::list<::Tags::IsUsingTimeSteppingErrorControl<>,
                     ::Tags::TimeStepper<LtsTimeStepper>, ::Tags::StepChoosers,
                     ::Tags::StepController, Tags::InitialTime>,
          tmpl::list<::Tags::NeverUsingTimeSteppingErrorControl,
                     ::Tags::TimeStepper<TimeStepper>, Tags::InitialTime>>>>;

  using simple_tags =
      tmpl::push_back<StepChoosers::step_chooser_simple_tags<Metavariables>,
                      ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                      ::Tags::Time, ::Tags::TimeStep,
                      ::Tags::Next<::Tags::TimeStep>>;
  using compute_tags = tmpl::list<::Tags::SubstepTimeCompute>;

  template <
      typename DbTagsList, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
                   typename db::DataBox<DbTagsList>::mutable_item_tags,
                   Initialization::Tags::InitialTime> and

               tmpl::list_contains_v<
                   typename db::DataBox<DbTagsList>::mutable_item_tags,
                   Initialization::Tags::InitialTimeDelta> and

               tmpl::list_contains_v<
                   typename db::DataBox<DbTagsList>::mutable_item_tags,
                   Tags::InitialSlabSize<Metavariables::local_time_stepping>>> =
          nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    const double initial_time_value = db::get<Tags::InitialTime>(box);
    const double initial_dt_value = db::get<Tags::InitialTimeDelta>(box);
    const double initial_slab_size =
        db::get<Tags::InitialSlabSize<Metavariables::local_time_stepping>>(box);

    const bool time_runs_forward = initial_dt_value > 0.0;
    const Slab initial_slab =
        time_runs_forward
            ? Slab::with_duration_from_start(initial_time_value,
                                             initial_slab_size)
            : Slab::with_duration_to_end(initial_time_value, initial_slab_size);
    const Time initial_time =
        time_runs_forward ? initial_slab.start() : initial_slab.end();
    const TimeDelta initial_dt =
        Evolution_detail::get_initial_time_step<Metavariables>(
            initial_time, initial_dt_value, box);

    // The slab number is increased in the self-start phase each
    // time one order of accuracy is obtained, and the evolution
    // proper starts with slab 0.
    const auto& time_stepper = db::get<::Tags::TimeStepper<>>(box);

    const TimeStepId time_id(
        time_runs_forward,
        -static_cast<int64_t>(time_stepper.number_of_past_steps()),
        initial_time);

    Initialization::mutate_assign<tmpl::list<
        ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>, ::Tags::Time,
        ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>>>(
        make_not_null(&box), TimeStepId{}, time_id,
        std::numeric_limits<double>::signaling_NaN(), initial_dt, initial_dt);
    return std::make_tuple(std::move(box));
  }

  template <
      typename DbTagsList, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<
          not(tmpl::list_contains_v<
                  typename db::DataBox<DbTagsList>::mutable_item_tags,
                  Initialization::Tags::InitialTime> and

              tmpl::list_contains_v<
                  typename db::DataBox<DbTagsList>::mutable_item_tags,
                  Initialization::Tags::InitialTimeDelta> and

              tmpl::list_contains_v<
                  typename db::DataBox<DbTagsList>::mutable_item_tags,
                  Tags::InitialSlabSize<Metavariables::local_time_stepping>>)> =
          nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    ERROR(
        "Could not find dependency 'Initialization::Tags::InitialTime', "
        "'Initialization::Tags::InitialTimeDelta', or "
        "'Tags::InitialSlabSize<Metavariables::local_time_stepping>' in "
        "DataBox.");
  }
};

/// \ingroup InitializationGroup
/// \brief Initialize time-stepper items
///
/// DataBox changes:
/// - Adds:
///   * `db::add_tag_prefix<Tags::dt, variables_tag>`
///   * `Tags::StepperError<variables_tag>`
///   * `Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>`
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note HistoryEvolvedVariables is allocated, but needs to be initialized
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <typename Metavariables>
struct TimeStepperHistory {
  static constexpr size_t dim = Metavariables::volume_dim;
  using variables_tag = typename Metavariables::system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using error_variables_tag =
      db::add_tag_prefix<::Tags::StepperError, variables_tag>;

  using simple_tags =
      tmpl::list<dt_variables_tag,
                 ::Tags::HistoryEvolvedVariables<variables_tag>,
                 error_variables_tag, ::Tags::StepperErrorUpdated>;

  using compute_tags = db::AddComputeTags<>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                         typename db::DataBox<DbTagsList>::mutable_item_tags,
                         domain::Tags::Mesh<dim>> and

                     tmpl::list_contains_v<
                         typename db::DataBox<DbTagsList>::mutable_item_tags,
                         variables_tag>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    using DtVars = typename dt_variables_tag::type;
    using ErrorVars = typename error_variables_tag::type;

    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();

    const auto& time_stepper = db::get<::Tags::TimeStepper<>>(box);
    const size_t starting_order =
        time_stepper.number_of_past_steps() == 0 ? time_stepper.order() : 1;
    // Will be overwritten before use
    DtVars dt_vars{num_grid_points};
    typename ::Tags::HistoryEvolvedVariables<variables_tag>::type history{
      starting_order};
    ErrorVars error_vars;
    // only bother allocating if the error vars are going to be used
    if (db::get<::Tags::IsUsingTimeSteppingErrorControlBase>(box)) {
      error_vars = ErrorVars{num_grid_points};
    }

    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(dt_vars), std::move(history),
        std::move(error_vars), false);

    return std::make_tuple(std::move(box));
  }

  template <
      typename DbTagsList, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<not(tmpl::list_contains_v<
                       typename db::DataBox<DbTagsList>::mutable_item_tags,
                       domain::Tags::Mesh<dim>> and

                   tmpl::list_contains_v<
                       typename db::DataBox<DbTagsList>::mutable_item_tags,
                       variables_tag>)> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    ERROR(
        "Could not find dependency '::Tags::Mesh<dim>' or "
        "'Metavariables::system::variables_tag' in DataBox.");
  }
};
}  // namespace Actions
}  // namespace Initialization
