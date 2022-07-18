// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta.hpp"

#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Math.hpp"

namespace TimeSteppers {

uint64_t RungeKutta::number_of_substeps() const {
  return butcher_tableau().result_coefficients.size();
}

uint64_t RungeKutta::number_of_substeps_for_error() const {
  return error_tableau().error_coefficients.size();
}

size_t RungeKutta::number_of_past_steps() const { return 0; }

namespace {
TimeStepId next_time_id_with_tableau(
    const TimeStepId& current_id, const TimeDelta& time_step,
    const RungeKutta::ButcherTableau& tableau) {
  const auto number_of_substeps = tableau.result_coefficients.size();
  const auto substep = current_id.substep();
  const auto step_time = current_id.step_time();
  const auto substep_time = current_id.substep_time();
  ASSERT(tableau.substep_times.size() == number_of_substeps - 1,
         "There should be one substep time for each substep, excluding the "
         "initial substep at 0.");

  if (substep == 0) {
    ASSERT(substep_time == step_time,
           "The first substep time should equal the step time "
               << step_time << ", not " << substep_time);
  } else {
    ASSERT(substep_time ==
               step_time + tableau.substep_times[substep - 1] * time_step,
           "The time for substep "
               << substep << " with step time " << step_time
               << " and time step " << time_step << " should be "
               << step_time + tableau.substep_times[substep - 1] * time_step
               << ", not " << substep_time);
  }

  if (substep >= number_of_substeps) {
    ERROR("In substep should be less than the number of steps, not "
          << substep << "/" << number_of_substeps);
  } else if (substep == number_of_substeps - 1) {
    return {current_id.time_runs_forward(), current_id.slab_number(),
            step_time + time_step};
  } else {
    return {current_id.time_runs_forward(), current_id.slab_number(), step_time,
            substep + 1,
            step_time + tableau.substep_times[substep] * time_step};
  }
}
}  // namespace

TimeStepId RungeKutta::next_time_id(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const {
  return next_time_id_with_tableau(current_id, time_step, butcher_tableau());
}

TimeStepId RungeKutta::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  const auto& tableau = error_tableau();
  const auto number_of_substeps = tableau.result_coefficients.size();
  const auto number_of_substeps_for_error = tableau.error_coefficients.size();

  const auto substep = current_id.substep();
  const auto step_time = current_id.step_time();

  if (number_of_substeps_for_error == number_of_substeps) {
    return next_time_id_with_tableau(current_id, time_step, tableau);
  }

  if (substep == number_of_substeps) {
    return {current_id.time_runs_forward(), current_id.slab_number(),
            step_time + time_step};
  } else if (substep == number_of_substeps - 1) {
    // Fake FSAL step.
    return {current_id.time_runs_forward(), current_id.slab_number(), step_time,
            substep + 1, step_time + time_step};
  } else {
    return next_time_id_with_tableau(current_id, time_step, tableau);
  }
}

const RungeKutta::ButcherTableau& RungeKutta::error_tableau() const {
  ASSERT(not butcher_tableau().error_coefficients.empty(),
         "No embedded error method was given and error_tableau() was not "
         "implemented for this stepper.");
  return butcher_tableau();
}

namespace {
template <typename T>
void update_between_substeps(const gsl::not_null<T*> u,
                             const gsl::not_null<UntypedHistory<T>*> history,
                             const double dt,
                             const std::vector<double>& coeffs_last,
                             const std::vector<double>& coeffs_this) {
  ASSERT(coeffs_last.size() + 1 == coeffs_this.size(),
         "Unexpected coefficient vector sizes.");
  if (coeffs_this.back() != 0.0) {
    *u += coeffs_this.back() * dt * *(history->end() - 1).derivative();
  }
  // The input state of *u is the previous substep, but Butcher
  // tableaus are given in terms of the start of the full step, so we
  // have to undo the previous substep as well as take this one.
  for (size_t i = 0; i < coeffs_last.size(); ++i) {
    const double coef = coeffs_this[i] - coeffs_last[i];
    if (coef != 0.0) {
      *u += coef * dt * *(history->begin() + static_cast<int>(i)).derivative();
    }
  }
}

template <typename T>
void update_u_impl_with_tableau(const gsl::not_null<T*> u,
                                const gsl::not_null<UntypedHistory<T>*> history,
                                const TimeDelta& time_step,
                                const RungeKutta::ButcherTableau& tableau) {
  const auto number_of_substeps = tableau.result_coefficients.size();
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  const double dt = time_step.value();

  ASSERT(tableau.substep_coefficients.size() == number_of_substeps - 1,
         "Tableau size inconsistency.");
  ASSERT(number_of_substeps > 1,
         "Implementing Euler's method is not supported by RungeKutta.");

  *u = *history->untyped_most_recent_value();
  if (substep == 0) {
    ASSERT(tableau.substep_coefficients[0].size() == 1,
           "First substep should use one derivative.");
    *u += tableau.substep_coefficients[0][0] * dt *
          *history->begin().derivative();
  } else if (substep == number_of_substeps - 1) {
    update_between_substeps(u, history, dt,
                            tableau.substep_coefficients[substep - 1],
                            tableau.result_coefficients);
  } else if (substep < number_of_substeps - 1) {
    update_between_substeps(u, history, dt,
                            tableau.substep_coefficients[substep - 1],
                            tableau.substep_coefficients[substep]);
  } else {
    ERROR("Substep should be less than " << number_of_substeps << ", not "
                                         << substep);
  }
}
}  // namespace

template <typename T>
void RungeKutta::update_u_impl(const gsl::not_null<T*> u,
                               const gsl::not_null<UntypedHistory<T>*> history,
                               const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  return update_u_impl_with_tableau(u, history, time_step, butcher_tableau());
}

template <typename T>
bool RungeKutta::update_u_impl(const gsl::not_null<T*> u,
                               const gsl::not_null<T*> u_error,
                               const gsl::not_null<UntypedHistory<T>*> history,
                               const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << history->integration_order());

  const auto& tableau = error_tableau();
  const auto number_of_substeps = tableau.result_coefficients.size();
  const auto number_of_substeps_for_error = tableau.error_coefficients.size();

  ASSERT(number_of_substeps_for_error == number_of_substeps or
             number_of_substeps_for_error == number_of_substeps + 1,
         "Number of error coefficients cannot exceed number of result "
         "coefficients by more than one (FSAL).  For extra substeps, "
         "implement error_tableau().");

  const size_t substep = (history->end() - 1).time_step_id().substep();

  // If we take a fake FSAL substep we don't actually update u.  It
  // would be nice to avoid the repeated RHS evaluation, but that
  // would require more extensive changes to the action loop.
  if (substep < number_of_substeps) {
    update_u_impl_with_tableau(u, history, time_step, tableau);
  } else {
    *u = *history->untyped_most_recent_value();
  }

  if (substep < number_of_substeps_for_error - 1) {
    return false;
  }

  const double dt = time_step.value();

  if (number_of_substeps_for_error != number_of_substeps) {
    // The time stepper uses FSAL
    *u_error = -tableau.error_coefficients.back() * dt *
               *(history->end() - 1).derivative();
  } else {
    // No term using the final value.
    *u_error = 0.0;
  }

  for (size_t i = 0; i < tableau.result_coefficients.size(); ++i) {
    const double coef =
        tableau.result_coefficients[i] - tableau.error_coefficients[i];
    if (coef != 0.0) {
      *u_error +=
          coef * dt * *(history->begin() + static_cast<int>(i)).derivative();
    }
  }

  return true;
}

template <typename T>
bool RungeKutta::dense_update_u_impl(const gsl::not_null<T*> u,
                                     const UntypedHistory<T>& history,
                                     const double time) const {
  if ((history.end() - 1).time_step_id().substep() != 0) {
    return false;
  }
  const double step_start = history[0].value();
  const double step_end = history[history.size() - 1].value();
  if (time == step_end) {
    // Special case necessary for dense output at the initial time,
    // before taking a step.
    *u = *history.untyped_most_recent_value();
    return true;
  }
  const evolution_less<double> before{step_end > step_start};
  if (history.size() == 1 or before(step_end, time)) {
    return false;
  }
  const double step_size = step_end - step_start;
  const double output_fraction = (time - step_start) / step_size;
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << step_start);

  // The interface doesn't tell us whether we've done an error
  // estimate, so we have to figure that out if it matters.
  bool using_error_tableau = false;
  if (&butcher_tableau() != &error_tableau()) {
    // In most cases the tableaus are the same and we never get here.
    const auto number_of_substeps =
        (history.end() - 2).time_step_id().substep() + 1;
    using_error_tableau =
        error_tableau().result_coefficients.size() == number_of_substeps;
    ASSERT(
        using_error_tableau or
            butcher_tableau().result_coefficients.size() == number_of_substeps,
        "Cannot determine which tableau to use for dense output.");
    ASSERT(
        not using_error_tableau or
            butcher_tableau().result_coefficients.size() != number_of_substeps,
        "Cannot determine which tableau to use for dense output.");
  }
  const auto& tableau =
      using_error_tableau ? error_tableau() : butcher_tableau();

  ASSERT(tableau.dense_coefficients.size() <= history.size(),
         "Insufficient history for dense output.  Most likely there are too "
         "many coefficient polynomials in the tableau.");
  *u = *history.untyped_most_recent_value();
  for (size_t i = 0; i < std::max(tableau.dense_coefficients.size(),
                                  tableau.result_coefficients.size());
       ++i) {
    const double coef =
        (i < tableau.dense_coefficients.size()
             ? evaluate_polynomial(tableau.dense_coefficients[i],
                                   output_fraction)
             : 0.0) -
        (i < tableau.result_coefficients.size() ? tableau.result_coefficients[i]
                                                : 0.0);
    if (coef != 0.0) {
      *u += coef * step_size *
            *(history.begin() + static_cast<int>(i)).derivative();
    }
  }

  return true;
}

template <typename T>
bool RungeKutta::can_change_step_size_impl(
    const TimeStepId& time_id, const UntypedHistory<T>& /*history*/) const {
  return time_id.substep() == 0;
}

TIME_STEPPER_DEFINE_OVERLOADS(RungeKutta)
}  // namespace TimeSteppers
