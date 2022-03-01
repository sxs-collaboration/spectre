// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Cerk5.hpp"

#include <cmath>
#include <limits>

#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {
Cerk5::Cerk5(CkMigrateMessage* /*msg*/) {}

size_t Cerk5::order() const { return 5; }

size_t Cerk5::error_estimate_order() const { return 4; }

uint64_t Cerk5::number_of_substeps() const { return 7; }

uint64_t Cerk5::number_of_substeps_for_error() const { return 7; }

size_t Cerk5::number_of_past_steps() const { return 0; }

// The stability polynomial is
//
//   p(z) = \sum_{n=0}^{stages-1} alpha_n z^n / n!,
//
// alpha_n=1.0 for n=1...(order-1). For the fifth order method:
//  alpha_6 = 6 (-5 c3**2 + 2 c3) - 2 c6 beta
//  alpha_7 = 14 c3 c6 beta
// where
//   beta = 20 c3**2 - 15 c3 + 3
// The stability limit as compared to a forward Euler method is given by finding
// the root for |p(-2 z)|-1=0. For forward Euler this is 1.0.
double Cerk5::stable_step() const { return 1.5961737362090775; }

TimeStepId Cerk5::next_time_id(const TimeStepId& current_id,
                               const TimeDelta& time_step) const {
  const auto& step = current_id.substep();
  const auto& t0 = current_id.step_time();
  const auto& t = current_id.substep_time();
  if (step < number_of_substeps()) {
    if (step == 0) {
      ASSERT(t == t0, "In CERK5 substep 0, the substep time ("
                          << t << ") should equal t0 (" << t0 << ")");
    } else {
      ASSERT(t == t0 + gsl::at(c_, step - 1) * time_step,
             "In CERK5 substep "
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
    ERROR("In CERK5 substep should be one of 0,1,2,3,4,5,6,7, not "
          << current_id.substep());
  }
}

TimeStepId Cerk5::next_time_id_for_error(const TimeStepId& current_id,
                                         const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

template <typename T>
void Cerk5::update_u_impl(const gsl::not_null<T*> u,
                          const gsl::not_null<UntypedHistory<T>*> history,
                          const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 5,
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
    case 2: {
      *u = *u0 + ((a4_[0] - a3_[0]) * dt) * *history->begin().derivative() +
           ((a4_[1] - a3_[1]) * dt) * *(history->begin() + 1).derivative() +
           (a4_[2] * dt) * *(history->begin() + 2).derivative();
      break;
    }
    case 3: {
      *u = *u0 + ((a5_[0] - a4_[0]) * dt) * *history->begin().derivative() +
           ((a5_[1] - a4_[1]) * dt) * *(history->begin() + 1).derivative() +
           ((a5_[2] - a4_[2]) * dt) * *(history->begin() + 2).derivative() +
           (a5_[3] * dt) * *(history->begin() + 3).derivative();
      break;
    }
    case 4: {
      *u = *u0 + ((a6_[0] - a5_[0]) * dt) * *history->begin().derivative() +
           ((a6_[1] - a5_[1]) * dt) * *(history->begin() + 1).derivative() +
           ((a6_[2] - a5_[2]) * dt) * *(history->begin() + 2).derivative() +
           ((a6_[3] - a5_[3]) * dt) * *(history->begin() + 3).derivative() +
           (a6_[4] * dt) * *(history->begin() + 4).derivative();
      break;
    }
    case 5: {
      *u = *u0 + ((a7_[0] - a6_[0]) * dt) * *history->begin().derivative() +
           ((a7_[1] - a6_[1]) * dt) * *(history->begin() + 1).derivative() +
           ((a7_[2] - a6_[2]) * dt) * *(history->begin() + 2).derivative() +
           ((a7_[3] - a6_[3]) * dt) * *(history->begin() + 3).derivative() +
           ((a7_[4] - a6_[4]) * dt) * *(history->begin() + 4).derivative() +
           (a7_[5] * dt) * *(history->begin() + 5).derivative();
      break;
    }
    case 6: {
      *u = *u0 + ((a8_[0] - a7_[0]) * dt) * *history->begin().derivative() +
           ((a8_[1] - a7_[1]) * dt) * *(history->begin() + 1).derivative() +
           ((a8_[2] - a7_[2]) * dt) * *(history->begin() + 2).derivative() +
           ((a8_[3] - a7_[3]) * dt) * *(history->begin() + 3).derivative() +
           ((a8_[4] - a7_[4]) * dt) * *(history->begin() + 4).derivative() +
           ((a8_[5] - a7_[5]) * dt) * *(history->begin() + 5).derivative() +
           (a8_[6] * dt) * *(history->begin() + 6).derivative();
      break;
    }
    default:
      ERROR("Bad substep value in CERK5: " << substep);
  }
}

template <typename T>
bool Cerk5::update_u_impl(const gsl::not_null<T*> u,
                          const gsl::not_null<T*> u_error,
                          const gsl::not_null<UntypedHistory<T>*> history,
                          const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 5,
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  update_u_impl(u, history, time_step);
  const size_t current_substep = (history->end() - 1).time_step_id().substep();
  if (current_substep == 6) {
    const double dt = time_step.value();
    *u_error = ((e_[0] - a8_[0]) * dt) * *history->begin().derivative() +
               ((e_[1] - a8_[1]) * dt) * *(history->begin() + 1).derivative() +
               ((e_[2] - a8_[2]) * dt) * *(history->begin() + 2).derivative() +
               ((e_[3] - a8_[3]) * dt) * *(history->begin() + 3).derivative() +
               ((e_[4] - a8_[4]) * dt) * *(history->begin() + 4).derivative() +
               ((e_[5] - a8_[5]) * dt) * *(history->begin() + 5).derivative() -
               a8_[6] * dt * *(history->begin() + 6).derivative();
    return true;
  }
  return false;
}

template <typename T>
bool Cerk5::dense_update_u_impl(const gsl::not_null<T*> u,
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

  // We need the following: k1, k2, k3, k4, k5, k6, k7, k8
  const auto k1 = history.begin().derivative();
  const auto k2 = (history.begin() + 1).derivative();
  const auto k3 = (history.begin() + 2).derivative();
  const auto k4 = (history.begin() + 3).derivative();
  const auto k5 = (history.begin() + 4).derivative();
  const auto k6 = (history.begin() + 5).derivative();
  const auto k7 = (history.begin() + 6).derivative();
  const auto k8 = (history.begin() + 7).derivative();

  *u = *u_n_plus_1 + (dt * evaluate_polynomial(b1_, output_fraction)) * *k1 +
       (dt * b2_) * *k2 +  //
       (dt * evaluate_polynomial(b3_, output_fraction)) * *k3 +
       (dt * evaluate_polynomial(b4_, output_fraction)) * *k4 +
       (dt * evaluate_polynomial(b5_, output_fraction)) * *k5 +
       (dt * evaluate_polynomial(b6_, output_fraction)) * *k6 +
       (dt * evaluate_polynomial(b7_, output_fraction)) * *k7 +
       (dt * evaluate_polynomial(b8_, output_fraction)) * *k8;
  return true;
}

template <typename T>
bool Cerk5::can_change_step_size_impl(
    const TimeStepId& time_id, const UntypedHistory<T>& /*history*/) const {
  return time_id.substep() == 0;
}

// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr double Cerk5::a2_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 2> Cerk5::a3_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 3> Cerk5::a4_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 4> Cerk5::a5_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 5> Cerk5::a6_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk5::a7_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 7> Cerk5::a8_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk5::b1_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr double Cerk5::b2_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk5::b3_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk5::b4_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk5::b5_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk5::b6_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk5::b7_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 6> Cerk5::b8_;
// NOLINTNEXTLINE(readability-redundant-declaration)
constexpr std::array<double, 8> Cerk5::e_;
const std::array<Time::rational_t, 6> Cerk5::c_ = {
    {{1, 6}, {1, 4}, {1, 2}, {1, 2}, {9, 14}, {7, 8}}};

TIME_STEPPER_DEFINE_OVERLOADS(Cerk5)
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Cerk5::my_PUP_ID =  // NOLINT
    0;
