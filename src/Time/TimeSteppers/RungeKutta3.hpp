// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class RungeKutta3.

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct TimeStepId;
namespace TimeSteppers {
template <typename Vars, typename DerivVars>
class History;
}  // namespace TimeSteppers
/// \endcond

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
///
/// A "strong stability-preserving" 3rd-order Runge-Kutta
/// time-stepper, as described in \cite HesthavenWarburton section
/// 5.7.
///
/// The CFL factor/stable step size is 1.25637266330916.
class RungeKutta3 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A third-order strong stability-preserving Runge-Kutta time-stepper."};

  RungeKutta3() = default;
  RungeKutta3(const RungeKutta3&) noexcept = default;
  RungeKutta3& operator=(const RungeKutta3&) noexcept = default;
  RungeKutta3(RungeKutta3&&) noexcept = default;
  RungeKutta3& operator=(RungeKutta3&&) noexcept = default;
  ~RungeKutta3() noexcept override = default;

  template <typename Vars, typename DerivVars>
  void update_u(gsl::not_null<Vars*> u,
                gsl::not_null<History<Vars, DerivVars>*> history,
                const TimeDelta& time_step) const noexcept;

  template <typename Vars, typename ErrVars, typename DerivVars>
  bool update_u(gsl::not_null<Vars*> u, gsl::not_null<ErrVars*> u_error,
                gsl::not_null<History<Vars, DerivVars>*> history,
                const TimeDelta& time_step) const noexcept;

  template <typename Vars, typename DerivVars>
  bool dense_update_u(gsl::not_null<Vars*> u,
                      const History<Vars, DerivVars>& history,
                      double time) const noexcept;

  size_t order() const noexcept override;

  size_t error_estimate_order() const noexcept override;

  uint64_t number_of_substeps() const noexcept override;

  uint64_t number_of_substeps_for_error() const noexcept override;

  size_t number_of_past_steps() const noexcept override;

  double stable_step() const noexcept override;

  TimeStepId next_time_id(const TimeStepId& current_id,
                          const TimeDelta& time_step) const noexcept override;

  TimeStepId next_time_id_for_error(
      const TimeStepId& current_id,
      const TimeDelta& time_step) const noexcept override;

  template <typename Vars, typename DerivVars>
  bool can_change_step_size(
      const TimeStepId& time_id,
      const TimeSteppers::History<Vars, DerivVars>& /*history*/) const
      noexcept {
    return time_id.substep() == 0;
  }

  WRAPPED_PUPable_decl_template(RungeKutta3);  // NOLINT

  explicit RungeKutta3(CkMigrateMessage* /*unused*/) noexcept {}

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& p) noexcept override {  // NOLINT
    TimeStepper::Inherit::pup(p);
  }
};

inline bool constexpr operator==(const RungeKutta3& /*lhs*/,
                                 const RungeKutta3& /*rhs*/) noexcept {
  return true;
}

inline bool constexpr operator!=(const RungeKutta3& /*lhs*/,
                                 const RungeKutta3& /*rhs*/) noexcept {
  return false;
}

template <typename Vars, typename DerivVars>
void RungeKutta3::update_u(
    const gsl::not_null<Vars*> u,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(history->integration_order() == 3,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  switch (substep) {
    case 0: {
      // from (5.32) of Hesthaven
      // v^(1) = u^n + dt*RHS(u^n,t^n)
      *u = history->most_recent_value() +
           time_step.value() * history->begin().derivative();
      break;
    }
    case 1: {
      // from (5.32) of Hesthaven
      // v^(2) = (1/4)*( 3*u^n + v^(1) + dt*RHS(v^(1),t^n + dt) )
      *u = history->most_recent_value() -
           0.25 * time_step.value() *
               (3.0 * history->begin().derivative() -
                (history->begin() + 1).derivative());
      break;
    }
    case 2: {
      // from (5.32) of Hesthaven
      // u^(n+1) = (1/3)*( u^n + 2*v^(2) + 2*dt*RHS(v^(2),t^n + (1/2)*dt) )
      *u = history->most_recent_value() -
           (1.0 / 12.0) * time_step.value() *
               (history->begin().derivative() +
                (history->begin() + 1).derivative() -
                8.0 * (history->begin() + 2).derivative());
      break;
    }
    default:
      ERROR("Bad substep value in RK3: " << substep);
  }
}

template <typename Vars, typename ErrVars, typename DerivVars>
bool RungeKutta3::update_u(
    const gsl::not_null<Vars*> u, const gsl::not_null<ErrVars*> u_error,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(history->integration_order() == 3,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  update_u(u, history, time_step);
  // error estimate is only available when completing a full step
  if ((history->end() - 1).time_step_id().substep() == 2) {
    // error is estimated by comparing the order 3 step result with an order 2
    // estimate. See e.g. Chapter II.4 of Harrier, Norsett, and Wagner 1993
    *u_error =
        -(1.0 / 3.0) * time_step.value() *
        (history->begin().derivative() + (history->begin() + 1).derivative() -
         2.0 * (history->begin() + 2).derivative());
    return true;
  }
  return false;
}

template <typename Vars, typename DerivVars>
bool RungeKutta3::dense_update_u(gsl::not_null<Vars*> u,
                                 const History<Vars, DerivVars>& history,
                                 const double time) const noexcept {
  if ((history.end() - 1).time_step_id().substep() != 0) {
    return false;
  }
  const double step_start = history.front().value();
  const double step_end = history.back().value();
  if (time == step_end) {
    // Special case necessary for dense output at the initial time,
    // before taking a step.
    *u = history.most_recent_value();
    return true;
  }
  const evolution_less<double> before{step_end > step_start};
  if (history.size() == 1 or before(step_end, time)) {
    return false;
  }
  const double time_step = step_end - step_start;
  const double output_fraction = (time - step_start) / time_step;
  ASSERT(output_fraction >= 0, "Attempting dense output at time " << time
         << ", but already progressed past " << step_start);
  ASSERT(output_fraction <= 1,
         "Requested time (" << time << " not within step [" << step_start
         << ", " << step_end << "]");

  // arXiv:1605.02429
  *u = history.most_recent_value() -
       (1.0 / 6.0) * time_step * (1.0 - output_fraction) *
           ((1.0 - 5.0 * output_fraction) * history.begin().derivative() +
            (1.0 + output_fraction) *
                ((history.begin() + 1).derivative() +
                 4.0 * (history.begin() + 2).derivative()));
  return true;
}
}  // namespace TimeSteppers
