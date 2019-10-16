// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Evolution_detail {
// Global time stepping
template <typename Metavariables,
          Requires<not Metavariables::local_time_stepping> = nullptr>
TimeDelta get_initial_time_step(
    const Time& initial_time, const double initial_dt_value,
    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
  return (initial_dt_value > 0.0 ? 1 : -1) * initial_time.slab().duration();
}

// Local time stepping
template <typename Metavariables,
          Requires<Metavariables::local_time_stepping> = nullptr>
TimeDelta get_initial_time_step(
    const Time& initial_time, const double initial_dt_value,
    const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
  const auto& step_controller = Parallel::get<Tags::StepController>(cache);
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
template <typename Metavariables>
struct TimeAndTimeStep {
  using initialization_tags =
      tmpl::list<Tags::InitialTime, Tags::InitialTimeDelta,
                 Tags::InitialSlabSize<Metavariables::local_time_stepping>>;

  template <
      typename DbTagsList, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
                   typename db::DataBox<DbTagsList>::simple_item_tags,
                   Initialization::Tags::InitialTime> and

               tmpl::list_contains_v<
                   typename db::DataBox<DbTagsList>::simple_item_tags,
                   Initialization::Tags::InitialTimeDelta> and

               tmpl::list_contains_v<
                   typename db::DataBox<DbTagsList>::simple_item_tags,
                   Tags::InitialSlabSize<Metavariables::local_time_stepping>>> =
          nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
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
    const TimeDelta initial_dt = Evolution_detail::get_initial_time_step(
        initial_time, initial_dt_value, cache);

    // The slab number is increased in the self-start phase each
    // time one order of accuracy is obtained, and the evolution
    // proper starts with slab 0.
    const auto& time_stepper = db::get<::Tags::TimeStepper<>>(box);

    const TimeStepId time_id(
        time_runs_forward,
        -static_cast<int64_t>(time_stepper.number_of_past_steps()),
        initial_time);

    using compute_tags = db::AddComputeTags<::Tags::SubstepTimeCompute>;

    return std::make_tuple(
        merge_into_databox<TimeAndTimeStep,
                           db::AddSimpleTags<::Tags::TimeStepId,
                                             ::Tags::Next<::Tags::TimeStepId>,
                                             ::Tags::Time, ::Tags::TimeStep>,
                           compute_tags>(
            std::move(box),
            // At this point we have not started evolution yet, so the current
            // time is undefined and _next_ is the initial time.
            TimeStepId{}, time_id, std::numeric_limits<double>::signaling_NaN(),
            initial_dt));
  }

  template <
      typename DbTagsList, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<
          not(tmpl::list_contains_v<
                  typename db::DataBox<DbTagsList>::simple_item_tags,
                  Initialization::Tags::InitialTime> and

              tmpl::list_contains_v<
                  typename db::DataBox<DbTagsList>::simple_item_tags,
                  Initialization::Tags::InitialTimeDelta> and

              tmpl::list_contains_v<
                  typename db::DataBox<DbTagsList>::simple_item_tags,
                  Tags::InitialSlabSize<Metavariables::local_time_stepping>>)> =
          nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
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
///   * `Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>`
///   * Tags::ComputeDeriv  (for non-conservative systems)
///   * Tags::ComputeDiv (for conservative systems)
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note HistoryEvolvedVariables is allocated, but needs to be initialized
template <typename Metavariables>
struct TimeStepperHistory {
  using initialization_tags = tmpl::list<>;

  static constexpr size_t dim = Metavariables::volume_dim;
  using variables_tag = typename Metavariables::system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  template <typename System, bool IsInFluxConservativeForm =
                                 System::is_in_flux_conservative_form>
  struct ComputeTags {
    using type = db::AddComputeTags<::Tags::DerivCompute<
        variables_tag,
        ::Tags::InverseJacobian<::Tags::ElementMap<dim>,
                                ::Tags::Coordinates<dim, Frame::Logical>>,
        typename System::gradients_tags>>;
  };

  template <typename System>
  struct ComputeTags<System, true> {
    using type = db::AddComputeTags<::Tags::DivCompute<
        db::add_tag_prefix<::Tags::Flux, variables_tag, tmpl::size_t<dim>,
                           Frame::Inertial>,
        ::Tags::InverseJacobian<::Tags::ElementMap<dim>,
                                ::Tags::Coordinates<dim, Frame::Logical>>>>;
  };

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                         typename db::DataBox<DbTagsList>::simple_item_tags,
                         ::Tags::Mesh<dim>> and

                     tmpl::list_contains_v<
                         typename db::DataBox<DbTagsList>::simple_item_tags,
                         variables_tag>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using DtVars = typename dt_variables_tag::type;

    const size_t num_grid_points =
        db::get<::Tags::Mesh<dim>>(box).number_of_grid_points();

    // Will be overwritten before use
    DtVars dt_vars{num_grid_points};
    typename ::Tags::HistoryEvolvedVariables<variables_tag>::type history;

    using compute_tags =
        typename ComputeTags<typename Metavariables::system>::type;
    return std::make_tuple(
        merge_into_databox<
            TimeStepperHistory,
            db::AddSimpleTags<dt_variables_tag,
                              ::Tags::HistoryEvolvedVariables<variables_tag>>,
            compute_tags>(std::move(box), std::move(dt_vars),
                          std::move(history)));
  }

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<not(tmpl::list_contains_v<
                             typename db::DataBox<DbTagsList>::simple_item_tags,
                             ::Tags::Mesh<dim>> and

                         tmpl::list_contains_v<
                             typename db::DataBox<DbTagsList>::simple_item_tags,
                             variables_tag>)> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Could not find dependency '::Tags::Mesh<dim>' or "
        "'Metavariables::system::variables_tag' in DataBox.");
  }
};
}  // namespace Actions
}  // namespace Initialization
