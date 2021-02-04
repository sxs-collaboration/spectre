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

  template <typename Vars, typename DerivVars>
  bool update_u(gsl::not_null<Vars*> u, gsl::not_null<Vars*> u_error,
                gsl::not_null<History<Vars, DerivVars>*> history,
                const TimeDelta& time_step) const noexcept;

  template <typename Vars, typename DerivVars>
  void dense_update_u(gsl::not_null<Vars*> u,
                      const History<Vars, DerivVars>& history,
                      double time) const noexcept;

  size_t order() const noexcept override;

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
  const size_t substep = history->size() - 1;
  const auto& vars = (history->end() - 1).value();
  const auto& dt_vars = (history->end() - 1).derivative();
  const auto& U0 = history->begin().value();

  switch (substep) {
    case 0: {
      // from (5.32) of Hesthaven
      // v^(1) = u^n + dt*RHS(u^n,t^n)
      // On entry V = u^n, U0 = u^n, rhs0 = RHS(u^n,t^n),
      // time = t^n
      *u = (history->end() - 1).value() + time_step.value() * dt_vars;
      // On exit v = v^(1), time = t^n + dt
      break;
    }
    case 1: {
      // from (5.32) of Hesthaven
      // v^(2) = (1/4)*( 3*u^n + v^(1) + dt*RHS(v^(1),t^n + dt) )
      // On entry V = v^(1), U0 = u^n, rhs0 = RHS(v^(1),t^n + dt),
      // time = t^n + dt
      *u = (history->end() - 1).value() +
           0.25 * (3.0 * (U0 - vars) + time_step.value() * dt_vars);
      // On exit v = v^(2), time = t^n + (1/2)*dt
      break;
    }
    case 2: {
      // from (5.32) of Hesthaven
      // u^(n+1) = (1/3)*( u^n + 2*v^(2) + 2*dt*RHS(v^(2),t^n + (1/2)*dt) )
      // On entry V = v^(2), U0 = u^n, rhs0 = RHS(v^(2),t^n + (1/2)*dt),
      // time = t^n + (1/2)*dt
      *u = (history->end() - 1).value() +
           (1.0 / 3.0) * (U0 - vars + 2.0 * time_step.value() * dt_vars);
      // On exit v = u^(n+1), time = t^n + dt
      break;
    }
    default:
      ERROR("Bad substep value in RK3: " << substep);
  }

  // Clean up old history
  if (history->size() == number_of_substeps()) {
    history->mark_unneeded(history->end());
  }
}

template <typename Vars, typename DerivVars>
bool RungeKutta3::update_u(
    const gsl::not_null<Vars*> u, const gsl::not_null<Vars*> u_error,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  const size_t substep = history->size() - 1;
  // error estimate is only available when completing a full step
  if(substep != 2) {
    update_u(u, history, time_step);
  } else {
    const auto& vars = (history->end() - 1).value();
    const auto& dt_vars = (history->end() - 1).derivative();
    const auto& U0 = history->begin().value();
    // from (5.32) of Hesthaven
    // u^(n+1) = (1/3)*( u^n + 2*v^(2) + 2*dt*RHS(v^(2),t^n + (1/2)*dt) )
    // On entry V = v^(2), U0 = u^n, rhs0 = RHS(v^(2),t^n + (1/2)*dt),
    // time = t^n + (1/2)*dt
    *u = (history->end() - 1).value() +
         (1.0 / 3.0) * (U0 - vars + 2.0 * time_step.value() * dt_vars);
    // On exit v = u^(n+1), time = t^n + dt

    // error is estimated by comparing the order 3 step result with an order 2
    // estimate. See e.g. Chapter II.4 of Harrier, Norsett, and Wagner 1993
    *u_error = *u - U0 -
               time_step.value() * 0.5 *
                   (history->begin().derivative() +
                    (history->begin() + 1).derivative());
    // Clean up old history
    if (history->size() == number_of_substeps()) {
      history->mark_unneeded(history->end());
    }
  }
  return substep == 2;
}

template <typename Vars, typename DerivVars>
void RungeKutta3::dense_update_u(gsl::not_null<Vars*> u,
                                 const History<Vars, DerivVars>& history,
                                 const double time) const noexcept {
  ASSERT(history.size() == 3, "Can only dense output on last substep");
  const double step_start = history[0].value();
  const double step_end = history[1].value();
  const double time_step = step_end - step_start;
  const double output_fraction = (time - step_start) / time_step;
  ASSERT(output_fraction >= 0, "Attempting dense output at time " << time
         << ", but already progressed past " << step_start);
  ASSERT(output_fraction <= 1,
         "Requested time (" << time << " not within step [" << step_start
         << ", " << step_end << "]");

  // arXiv:1605.02429
  *u += (1.0 - output_fraction * (1.0 - output_fraction / 3.0)) *
            history.begin().value() +
        output_fraction * (1.0 - output_fraction) *
            (history.begin() + 1).value() +
        (2.0 / 3.0 * square(output_fraction) - 1.0) *
            (history.begin() + 2).value() +
        2.0 / 3.0 * square(output_fraction) * time_step *
            (history.begin() + 2).derivative();
}
}  // namespace TimeSteppers
