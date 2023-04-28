// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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
/// GlobalCache, MutableGlobalCache, and DataBox and how they are initialized
///
/// Since the evolution has not started yet, initialize the state
/// _before_ the initial time. So `Tags::TimeStepId` is undefined at this point,
/// and `Tags::Next<Tags::TimeStepId>` is the initial time.
template <typename Metavariables, bool UsingLts>
struct TimeStepping {
  using TimeStepperType =
      tmpl::conditional_t<UsingLts, LtsTimeStepper, TimeStepper>;

  /// Tags for constant items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using const_global_cache_tags = tmpl::list<>;

  /// Tags for mutable items added to the MutableGlobalCache.  These items are
  /// initialized from input file options.
  using mutable_global_cache_tags = tmpl::list<>;

  /// Tags for items fetched by the DataBox and passed to the apply function
  using argument_tags = tmpl::list<::Tags::Time, Tags::InitialTimeDelta,
                                   Tags::InitialSlabSize<UsingLts>,
                                   ::Tags::TimeStepper<TimeStepperType>>;

  /// Tags for simple DataBox items that are initialized from input file options
  using simple_tags_from_options = tmpl::flatten<
      tmpl::list<argument_tags,
                 tmpl::conditional_t<UsingLts, tmpl::list<::Tags::StepChoosers>,
                                     tmpl::list<>>>>;

  /// Tags for simple DataBox items that are default initialized.
  using default_initialized_simple_tags = tmpl::push_back<
      StepChoosers::step_chooser_simple_tags<Metavariables, UsingLts>,
      ::Tags::TimeStepId, ::Tags::AdaptiveSteppingDiagnostics>;

  /// Tags for items in the DataBox that are mutated by the apply function
  using return_tags =
      tmpl::list<::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
                 ::Tags::Next<::Tags::TimeStep>>;

  /// Tags for mutable DataBox items that are either default initialized or
  /// initialized by the apply function
  using simple_tags =
      tmpl::append<default_initialized_simple_tags, return_tags>;

  /// Tags for immutable DataBox items (compute items or reference items) added
  /// to the DataBox.
  using compute_tags = tmpl::list<>;

  /// Given the items fetched from a DataBox by the argument_tags when UsingLts
  /// is true, mutate the items in the DataBox corresponding to return_tags
  static void apply(const gsl::not_null<TimeStepId*> next_time_step_id,
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<TimeDelta*> next_time_step,
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
  }

  /// Given the items fetched from a DataBox by the argument_tags, when UsingLts
  /// is false, mutate the items in the DataBox corresponding to return_tags
  static void apply(const gsl::not_null<TimeStepId*> next_time_step_id,
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<TimeDelta*> next_time_step,
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
      tmpl::list<::Tags::TimeStepper<>, domain::Tags::Mesh<dim>>;
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
}  // namespace Initialization
