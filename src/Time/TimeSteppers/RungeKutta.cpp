// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/RungeKutta.hpp"

#include <algorithm>

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
  return std::max(butcher_tableau().result_coefficients.size(),
                  butcher_tableau().error_coefficients.size());
}

size_t RungeKutta::number_of_past_steps() const { return 0; }

namespace {
TimeStepId next_time_id_from_substeps(
    const TimeStepId& current_id, const TimeDelta& time_step,
    const std::vector<Rational>& substep_times,
    const size_t number_of_substeps) {
  ASSERT(substep_times.size() + 1 >= number_of_substeps,
         "More result coefficients than substeps");
  const auto substep = current_id.substep();
  const auto step_time = current_id.step_time();
  const auto substep_time = current_id.substep_time();

  if (substep == 0) {
    ASSERT(substep_time == step_time,
           "The first substep time should equal the step time "
               << step_time << ", not " << substep_time);
  } else {
    ASSERT(substep_time == step_time + substep_times[substep - 1] * time_step,
           "The time for substep "
               << substep << " with step time " << step_time
               << " and time step " << time_step << " should be "
               << step_time + substep_times[substep - 1] * time_step
               << ", not " << substep_time);
  }

  if (substep >= number_of_substeps) {
    ERROR("In substep should be less than the number of steps, not "
          << substep << "/" << number_of_substeps);
  } else if (substep == number_of_substeps - 1) {
    return current_id.next_step(time_step);
  } else {
    return current_id.next_substep(time_step, substep_times[substep]);
  }
}
}  // namespace

TimeStepId RungeKutta::next_time_id(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const {
  return next_time_id_from_substeps(current_id, time_step,
                                    butcher_tableau().substep_times,
                                    number_of_substeps());
}

TimeStepId RungeKutta::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  return next_time_id_from_substeps(current_id, time_step,
                                    butcher_tableau().substep_times,
                                    number_of_substeps_for_error());
}

namespace {
template <typename T>
void update_between_substeps(const gsl::not_null<T*> u,
                             const ConstUntypedHistory<T>& history,
                             const double dt,
                             const std::vector<double>& coeffs_last,
                             const std::vector<double>& coeffs_this) {
  const size_t number_of_substeps =
      std::max(coeffs_last.size(), coeffs_this.size());
  for (size_t i = 0; i < number_of_substeps; ++i) {
    double coef = 0.0;
    if (i < coeffs_this.size()) {
      coef += coeffs_this[i];
    }
    if (i < coeffs_last.size()) {
      coef -= coeffs_last[i];
    }
    if (coef != 0.0) {
      *u += coef * dt *
            (i == 0 ? history.front() : history.substeps()[i - 1]).derivative;
    }
  }
}

template <typename T>
void update_u_impl_with_tableau(const gsl::not_null<T*> u,
                                const MutableUntypedHistory<T>& history,
                                const TimeDelta& time_step,
                                const RungeKutta::ButcherTableau& tableau,
                                const size_t number_of_substeps) {
  // Clean up old history
  if (history.at_step_start()) {
    history.clear_substeps();
    if (history.size() > 1) {
      history.pop_front();
    }
    history.discard_value(history.back().time_step_id);
  } else {
    history.discard_value(history.substeps().back().time_step_id);
  }
  ASSERT(history.size() == 1, "Have more than one step after cleanup.");

  const double dt = time_step.value();

  ASSERT(number_of_substeps > 1,
         "Implementing Euler's method is not supported by RungeKutta.");

  const auto substep = history.substeps().size();
  if (substep == 0) {
    ASSERT(tableau.substep_coefficients[0].size() == 1,
           "First substep should use one derivative.");
    *u += tableau.substep_coefficients[0][0] * dt * history.front().derivative;
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
                               const MutableUntypedHistory<T>& history,
                               const TimeDelta& time_step) const {
  ASSERT(history.integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << history.integration_order());
  return update_u_impl_with_tableau(u, history, time_step, butcher_tableau(),
                                    number_of_substeps());
}

template <typename T>
bool RungeKutta::update_u_impl(const gsl::not_null<T*> u,
                               const gsl::not_null<T*> u_error,
                               const MutableUntypedHistory<T>& history,
                               const TimeDelta& time_step) const {
  ASSERT(history.integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << history.integration_order());

  const auto& tableau = butcher_tableau();
  const auto number_of_substeps = number_of_substeps_for_error();
  update_u_impl_with_tableau(u, history, time_step, tableau,
                             number_of_substeps);

  const size_t substep = history.substeps().size();

  if (substep < number_of_substeps - 1) {
    return false;
  }

  const double dt = time_step.value();
  *u_error = 0.0;
  update_between_substeps(u_error, history, dt, tableau.error_coefficients,
                          tableau.result_coefficients);

  return true;
}

template <typename T>
bool RungeKutta::dense_update_u_impl(const gsl::not_null<T*> u,
                                     const ConstUntypedHistory<T>& history,
                                     const double time) const {
  if (not history.at_step_start()) {
    return false;
  }
  const double step_start = history.front().time_step_id.step_time().value();
  const double step_end = history.back().time_step_id.step_time().value();
  if (time == step_end) {
    // Special case necessary for dense output at the initial time,
    // before taking a step.
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

  const auto& tableau = butcher_tableau();

  // The Butcher dense output coefficients are given in terms of the
  // start of the step, but we are passed the value at the end of the
  // step, so we have to undo the step first.
  for (size_t i = 0; i < tableau.result_coefficients.size(); ++i) {
    const double coef = tableau.result_coefficients[i];
    if (coef != 0.0) {
      *u -= coef * step_size *
            (i == 0 ? history.front() : history.substeps()[i - 1]).derivative;
    }
  }

  const auto number_of_dense_coefficients = tableau.dense_coefficients.size();
  const size_t number_of_substep_terms = std::min(
      tableau.result_coefficients.size(), number_of_dense_coefficients);
  for (size_t i = 0; i < number_of_substep_terms; ++i) {
    const double coef =
        evaluate_polynomial(tableau.dense_coefficients[i], output_fraction);
    if (coef != 0.0) {
      *u += coef * step_size *
            (i == 0 ? history.front() : history.substeps()[i - 1]).derivative;
    }
  }

  if (number_of_dense_coefficients > number_of_substep_terms) {
    // We use the derivative at the end of the step.
    const double coef =
        evaluate_polynomial(tableau.dense_coefficients.back(), output_fraction);
    if (coef != 0.0) {
      *u += coef * step_size * history.back().derivative;
    }
  }

  return true;
}

template <typename T>
bool RungeKutta::can_change_step_size_impl(
    const TimeStepId& time_id,
    const ConstUntypedHistory<T>& /*history*/) const {
  return time_id.substep() == 0;
}

TIME_STEPPER_DEFINE_OVERLOADS(RungeKutta)
}  // namespace TimeSteppers
