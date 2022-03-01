// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Cerk2.hpp"

#include <cmath>
#include <limits>

#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

size_t Cerk2::order() const { return 2; }

size_t Cerk2::error_estimate_order() const { return 1; }

uint64_t Cerk2::number_of_substeps() const { return 2; }

uint64_t Cerk2::number_of_substeps_for_error() const { return 2; }

size_t Cerk2::number_of_past_steps() const { return 0; }

// The stability polynomial is
//
//   p(z) = \sum_{n=0}^{stages-1} alpha_n z^n / n!,
//
// alpha_n=1.0 for n=1...(order-1). It is the same as for forward Euler.
double Cerk2::stable_step() const { return 1.0; }

TimeStepId Cerk2::next_time_id(const TimeStepId& current_id,
                               const TimeDelta& time_step) const {
  const auto& step = current_id.substep();
  const auto& t0 = current_id.step_time();
  const auto& t = current_id.substep_time();
  if (step < number_of_substeps()) {
    if (step == 0) {
      ASSERT(t == t0, "In Cerk2 substep 0, the substep time ("
                          << t << ") should equal t0 (" << t0 << ")");
    } else {
      ASSERT(t == t0 + gsl::at(c_, step - 1) * time_step,
             "In Cerk2 substep "
                 << step << ", the substep time (" << t
                 << ") should equal t0+c[" << step - 1 << "]*dt ("
                 << t0 + gsl::at(c_, step - 1) * time_step << ")");
    }
    if (step < number_of_substeps() - 1) {
      return {current_id.time_runs_forward(), current_id.slab_number(), t0,
              step + 1, t0 + gsl::at(c_, step) * time_step};
    } else {
      return {current_id.time_runs_forward(), current_id.slab_number(),
              t0 + time_step};
    }
  } else {
    ERROR("In Cerk2 substep should be one of 0,1, not "
          << current_id.substep());
  }
}

TimeStepId Cerk2::next_time_id_for_error(const TimeStepId& current_id,
                                         const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

template <typename T>
void Cerk2::update_u_impl(const gsl::not_null<T*> u,
                          const gsl::not_null<UntypedHistory<T>*> history,
                          const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 2,
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  const auto u0 = history->untyped_most_recent_value();
  const double dt = time_step.value();

  switch (substep) {
    case 0: {
      *u = *u0 + (a2_ * dt) * *history->begin().derivative();
      break;
    }
    case 1: {
      *u = *u0 + ((a3_[0] - a2_) * dt) * *history->begin().derivative() +
           (a3_[1] * dt) * *(history->begin() + 1).derivative();
      break;
    }
    default:
      ERROR("Bad substep value in Cerk2: " << substep);
  }
}

template <typename T>
bool Cerk2::update_u_impl(const gsl::not_null<T*> u,
                          const gsl::not_null<T*> u_error,
                          const gsl::not_null<UntypedHistory<T>*> history,
                          const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 2,
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  update_u_impl(u, history, time_step);
  const size_t current_substep = (history->end() - 1).time_step_id().substep();
  if (current_substep == 1) {
    const double dt = time_step.value();
    *u_error = ((e_[0] - a3_[0]) * dt) * *history->begin().derivative() -
               a3_[1] * dt * *(history->begin() + 1).derivative();
    return true;
  }
  return false;
}

template <typename T>
bool Cerk2::dense_update_u_impl(const gsl::not_null<T*> u,
                                const UntypedHistory<T>& history,
                                const double time) const {
  if ((history.end() - 1).time_step_id().substep() != 0) {
    return false;
  }
  const double t0 = history[0].value();
  const double t_end = history[history.size() - 1].value();
  if (time == t_end) {
    // Special case necessary for dense output at the initial time,
    // before taking a step.
    *u = *history.untyped_most_recent_value();
    return true;
  }
  const evolution_less<double> before{t_end > t0};
  if (history.size() == 1 or before(t_end, time)) {
    return false;
  }
  const double dt = t_end - t0;
  const double output_fraction = (time - t0) / dt;
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << t0);
  ASSERT(output_fraction <= 1.0, "Requested time ("
                                     << time << ") not within step [" << t0
                                     << ", " << t0 + dt << "]");

  const auto u_n_plus_1 = history.untyped_most_recent_value();

  // We need the following: k1, k2
  const auto k1 = history.begin().derivative();
  const auto k2 = (history.begin() + 1).derivative();

  *u = *u_n_plus_1 + (dt * evaluate_polynomial(b1_, output_fraction)) * *k1 +
       (dt * evaluate_polynomial(b2_, output_fraction)) * *k2;
  return true;
}

template <typename T>
bool Cerk2::can_change_step_size_impl(
    const TimeStepId& time_id, const UntypedHistory<T>& /*history*/) const {
  return time_id.substep() == 0;
}

// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr double Cerk2::a2_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 2> Cerk2::a3_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 3> Cerk2::b1_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 3> Cerk2::b2_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 2> Cerk2::e_;
const std::array<Time::rational_t, 1> Cerk2::c_ = {{{1, 1}}};

TIME_STEPPER_DEFINE_OVERLOADS(Cerk2)
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Cerk2::my_PUP_ID =  // NOLINT
    0;
