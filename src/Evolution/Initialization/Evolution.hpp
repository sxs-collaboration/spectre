// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Parallel/Tags/ArrayIndex.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Initialization {

namespace detail {
inline Time initial_time(const bool time_runs_forward,
                         const double initial_time_value,
                         const double initial_slab_size) {
  const Slab initial_slab =
      time_runs_forward
          ? Slab::with_duration_from_start(initial_time_value,
                                           initial_slab_size)
          : Slab::with_duration_to_end(initial_time_value, initial_slab_size);
  return time_runs_forward ? initial_slab.start() : initial_slab.end();
}

template <typename TimeStepper>
void set_next_time_step_id(const gsl::not_null<TimeStepId*> next_time_step_id,
                           const Time& initial_time,
                           const bool time_runs_forward,
                           const TimeStepper& time_stepper) {
  *next_time_step_id = TimeStepId(
      time_runs_forward,
      -static_cast<int64_t>(time_stepper.number_of_past_steps()), initial_time);
}
}  // namespace detail

/// \ingroup InitializationGroup
/// \brief Initialize items related to time stepping
///
/// \details See the type aliases defined below for what items are added to the
/// GlobalCache and DataBox and how they are initialized
///
/// Since the evolution has not started yet, initialize the state
/// _before_ the initial time. So `Tags::TimeStepId` is undefined at this point,
/// and `Tags::Next<Tags::TimeStepId>` is the initial time.
template <typename Metavariables, typename TimeStepperBase>
struct TimeStepping {
  /// Tags for constant items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using const_global_cache_tags = tmpl::conditional_t<
      TimeStepperBase::local_time_stepping,
      tmpl::list<::Tags::ConcreteTimeStepper<TimeStepperBase>,
                 ::Tags::StepChoosers>,
      tmpl::list<::Tags::ConcreteTimeStepper<TimeStepperBase>>>;

  /// Tags for mutable items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using mutable_global_cache_tags = tmpl::list<>;

  /// Tags for items fetched by the DataBox and passed to the apply function
  using argument_tags =
      tmpl::list<::Tags::Time, Tags::InitialTimeDelta,
                 Tags::InitialSlabSize<TimeStepperBase::local_time_stepping>,
                 ::Tags::ConcreteTimeStepper<TimeStepperBase>>;

  /// Tags for simple DataBox items that are initialized from input file options
  using simple_tags_from_options =
      tmpl::list<::Tags::Time, Tags::InitialTimeDelta,
                 Tags::InitialSlabSize<TimeStepperBase::local_time_stepping>>;

  /// Tags for simple DataBox items that are default initialized.
  using default_initialized_simple_tags =
      tmpl::push_back<StepChoosers::step_chooser_simple_tags<
                          Metavariables, TimeStepperBase::local_time_stepping>,
                      ::Tags::TimeStepId, ::Tags::StepNumberWithinSlab,
                      ::Tags::AdaptiveSteppingDiagnostics>;

  /// Tags for items in the DataBox that are mutated by the apply function
  using return_tags =
      tmpl::list<::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
                 ::Tags::Next<::Tags::TimeStep>,
                 ::Tags::ChangeSlabSize::SlabSizeGoal>;

  /// Tags for mutable DataBox items that are either default initialized or
  /// initialized by the apply function
  using simple_tags =
      tmpl::append<default_initialized_simple_tags, return_tags>;

  /// Tags for immutable DataBox items (compute items or reference items) added
  /// to the DataBox.
  using compute_tags = time_stepper_ref_tags<TimeStepperBase>;

  /// Given the items fetched from a DataBox by the argument_tags when using
  /// LTS, mutate the items in the DataBox corresponding to return_tags
  static void apply(const gsl::not_null<TimeStepId*> next_time_step_id,
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<TimeDelta*> next_time_step,
                    const gsl::not_null<double*> slab_size_goal,
                    const double initial_time_value,
                    const double initial_dt_value,
                    const double initial_slab_size,
                    const LtsTimeStepper& time_stepper) {
    const bool time_runs_forward = initial_dt_value > 0.0;
    const Time initial_time = detail::initial_time(
        time_runs_forward, initial_time_value, initial_slab_size);
    detail::set_next_time_step_id(next_time_step_id, initial_time,
                                  time_runs_forward, time_stepper);
    *time_step = choose_lts_step_size(initial_time, initial_dt_value);
    *next_time_step = *time_step;
    *slab_size_goal =
        time_runs_forward ? initial_slab_size : -initial_slab_size;
  }

  /// Given the items fetched from a DataBox by the argument_tags, when not
  /// using LTS, mutate the items in the DataBox corresponding to return_tags
  static void apply(const gsl::not_null<TimeStepId*> next_time_step_id,
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<TimeDelta*> next_time_step,
                    const gsl::not_null<double*> slab_size_goal,
                    const double initial_time_value,
                    const double initial_dt_value,
                    const double initial_slab_size,
                    const TimeStepper& time_stepper) {
    const bool time_runs_forward = initial_dt_value > 0.0;
    const Time initial_time = detail::initial_time(
        time_runs_forward, initial_time_value, initial_slab_size);
    detail::set_next_time_step_id(next_time_step_id, initial_time,
                                  time_runs_forward, time_stepper);
    *time_step = (time_runs_forward ? 1 : -1) * initial_time.slab().duration();
    *next_time_step = *time_step;
    *slab_size_goal =
        time_runs_forward ? initial_slab_size : -initial_slab_size;
  }
};

/// \brief Initialize/update items related to time stepping after an AMR change
template <size_t Dim>
struct ProjectTimeStepping : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags =
      tmpl::list<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                 ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>, ::Tags::Time,
                 ::Tags::StepNumberWithinSlab,
                 ::Tags::AdaptiveSteppingDiagnostics,
                 ::Tags::ChangeSlabSize::SlabSizeGoal>;
  using argument_tags = tmpl::list<Parallel::Tags::ArrayIndex>;

  static void apply(
      const gsl::not_null<TimeStepId*> /*time_step_id*/,
      const gsl::not_null<TimeStepId*> /*next_time_step_id*/,
      const gsl::not_null<TimeDelta*> /*time_step*/,
      const gsl::not_null<TimeDelta*> /*next_time_step*/,
      const gsl::not_null<double*> /*time*/,
      const gsl::not_null<uint64_t*> /*step_number_within_slab*/,
      const gsl::not_null<AdaptiveSteppingDiagnostics*>
      /*adaptive_stepping_diagnostics*/,
      const gsl::not_null<double*> /*slab_size_goal*/,
      const ElementId<Dim>& /*element_id*/,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {
    // Do not change anything for p-refinement
  }

  template <typename... Tags>
  static void apply(const gsl::not_null<TimeStepId*> time_step_id,
                    const gsl::not_null<TimeStepId*> next_time_step_id,
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<TimeDelta*> next_time_step,
                    const gsl::not_null<double*> time,
                    const gsl::not_null<uint64_t*> step_number_within_slab,
                    const gsl::not_null<AdaptiveSteppingDiagnostics*>
                        adaptive_stepping_diagnostics,
                    const gsl::not_null<double*> slab_size_goal,
                    const ElementId<Dim>& element_id,
                    const tuples::TaggedTuple<Tags...>& parent_items) {
    *time_step_id = get<::Tags::TimeStepId>(parent_items);
    *next_time_step_id = get<::Tags::Next<::Tags::TimeStepId>>(parent_items);
    *time_step = get<::Tags::TimeStep>(parent_items);
    *next_time_step = get<::Tags::Next<::Tags::TimeStep>>(parent_items);
    *time = get<::Tags::Time>(parent_items);
    *slab_size_goal = get<::Tags::ChangeSlabSize::SlabSizeGoal>(parent_items);
    *step_number_within_slab = get<::Tags::StepNumberWithinSlab>(parent_items);

    // Since AdaptiveSteppingDiagnostics are reduced over all elements, we
    // set the slab quantities to the same value over all children, and the
    // step quantities to belong to the first child
    const auto& parent_diagnostics =
        get<::Tags::AdaptiveSteppingDiagnostics>(parent_items);
    const auto& parent_amr_flags =
        get<amr::Tags::Info<Dim>>(parent_items).flags;
    const auto& parent_id =
        get<Parallel::Tags::ArrayIndexImpl<ElementId<Dim>>>(parent_items);
    auto children_ids = amr::ids_of_children(parent_id, parent_amr_flags);
    if (element_id == children_ids.front()) {
      *adaptive_stepping_diagnostics = parent_diagnostics;
    } else {
      adaptive_stepping_diagnostics->number_of_slabs =
          parent_diagnostics.number_of_slabs;
      adaptive_stepping_diagnostics->number_of_slab_size_changes =
          parent_diagnostics.number_of_slab_size_changes;
    }
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<TimeStepId*> time_step_id,
      const gsl::not_null<TimeStepId*> next_time_step_id,
      const gsl::not_null<TimeDelta*> time_step,
      const gsl::not_null<TimeDelta*> next_time_step,
      const gsl::not_null<double*> time,
      const gsl::not_null<uint64_t*> step_number_within_slab,
      const gsl::not_null<AdaptiveSteppingDiagnostics*>
          adaptive_stepping_diagnostics,
      const gsl::not_null<double*> slab_size_goal,
      const ElementId<Dim>& /*element_id*/,
      const std::unordered_map<ElementId<Dim>, tuples::TaggedTuple<Tags...>>&
          children_items) {
    const auto slowest_child =
        alg::min_element(children_items, [](const auto& a, const auto& b) {
          const auto& time_step_a = get<::Tags::TimeStep>(a.second);
          const auto& time_step_b = get<::Tags::TimeStep>(b.second);
          ASSERT(time_step_a.is_positive() == time_step_b.is_positive(),
                 "Elements are not taking time steps in the same direction!");
          return time_step_a.is_positive() ? (time_step_a < time_step_b)
                                           : (time_step_a > time_step_b);
        });
    const auto& slowest_child_items = (*slowest_child).second;
    *time_step_id = get<::Tags::TimeStepId>(slowest_child_items);
    *next_time_step_id =
        get<::Tags::Next<::Tags::TimeStepId>>(slowest_child_items);
    *time_step = get<::Tags::TimeStep>(slowest_child_items);
    *next_time_step = get<::Tags::Next<::Tags::TimeStep>>(slowest_child_items);
    *time = get<::Tags::Time>(slowest_child_items);
    *slab_size_goal =
        get<::Tags::ChangeSlabSize::SlabSizeGoal>(slowest_child_items);
    *step_number_within_slab =
        get<::Tags::StepNumberWithinSlab>(slowest_child_items);
    const auto& slowest_child_diagnostics =
        get<::Tags::AdaptiveSteppingDiagnostics>(slowest_child_items);

    adaptive_stepping_diagnostics->number_of_slabs =
        slowest_child_diagnostics.number_of_slabs;
    adaptive_stepping_diagnostics->number_of_slab_size_changes =
        slowest_child_diagnostics.number_of_slab_size_changes;
    for (const auto& [_, child_items] : children_items) {
      *adaptive_stepping_diagnostics +=
          get<::Tags::AdaptiveSteppingDiagnostics>(child_items);
    }
  }
};

/// \ingroup InitializationGroup
/// \brief Initialize time-stepper items
///
/// DataBox changes:
/// - Adds:
///   * `db::add_tag_prefix<Tags::dt, variables_tag>`
///   * `Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>`
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note HistoryEvolvedVariables is allocated, but needs to be initialized
template <typename Metavariables>
struct TimeStepperHistory {
  static constexpr size_t dim = Metavariables::volume_dim;
  using variables_tag = typename Metavariables::system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags =
      tmpl::list<dt_variables_tag,
                 ::Tags::HistoryEvolvedVariables<variables_tag>>;
  using compute_tags = tmpl::list<>;

  using argument_tags =
      tmpl::list<::Tags::TimeStepper<TimeStepper>, domain::Tags::Mesh<dim>>;
  using return_tags = simple_tags;

  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> dt_vars,
      const gsl::not_null<TimeSteppers::History<typename variables_tag::type>*>
          history,
      const TimeStepper& time_stepper, const Mesh<dim>& mesh) {
    // Will be overwritten before use
    dt_vars->initialize(mesh.number_of_grid_points());

    // All steppers we have that need to start at low order require
    // one additional point per order, so this is the order that
    // requires no initial past steps.
    const size_t starting_order =
        time_stepper.order() - time_stepper.number_of_past_steps();
    history->integration_order(starting_order);
  }
};

/// \brief Initialize/update items related to time stepper history after an AMR
/// change
template <typename Metavariables>
struct ProjectTimeStepperHistory : tt::ConformsTo<amr::protocols::Projector> {
  static constexpr size_t dim = Metavariables::volume_dim;
  using variables_tag = typename Metavariables::system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using history_tag = ::Tags::HistoryEvolvedVariables<variables_tag>;

  using return_tags = tmpl::list<dt_variables_tag, history_tag>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<dim>, Parallel::Tags::ArrayIndex>;

  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> dt_vars,
      const gsl::not_null<typename history_tag::type*> history,
      const Mesh<dim>& new_mesh, const ElementId<dim>& /*element_id*/,
      const std::pair<Mesh<dim>, Element<dim>>& old_mesh_and_element) {
    const auto& old_mesh = old_mesh_and_element.first;
    if (old_mesh == new_mesh) {
      return;  // mesh was not refined, so no projection needed
    }
    const auto projection_matrices =
        Spectral::p_projection_matrices(old_mesh, new_mesh);
    const auto& old_extents = old_mesh.extents();
    history->map_entries(
        [&projection_matrices, &old_extents](const auto entry) {
          *entry = apply_matrices(projection_matrices, *entry, old_extents);
        });
    dt_vars->initialize(new_mesh.number_of_grid_points());
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> /*dt_vars*/,
      const gsl::not_null<typename history_tag::type*> /*history*/,
      const Mesh<dim>& /*new_mesh*/, const ElementId<dim>& /*element_id*/,
      const tuples::TaggedTuple<Tags...>& /*parent_items*/) {
    ERROR("h-refinement not implemented yet");
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> /*dt_vars*/,
      const gsl::not_null<typename history_tag::type*> /*history*/,
      const Mesh<dim>& /*new_mesh*/, const ElementId<dim>& /*element_id*/,
      const std::unordered_map<ElementId<dim>, tuples::TaggedTuple<Tags...>>&
      /*children_items*/) {
    ERROR("h-refinement not implemented yet");
  }
};
}  // namespace Initialization
