// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class RungeKutta4.

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

/*!
 * \ingroup TimeSteppersGroup
 *
 * The standard 4th-order Runge-Kutta method, given e.g. in
 * https://en.wikipedia.org/wiki/Runge-Kutta_methods
 * that solves equations of the form
 *
 * \f{eqnarray}{
 * \frac{du}{dt} & = & \mathcal{L}(t,u).
 * \f}
 * Given a solution \f$u(t^n)=u^n\f$, this stepper computes
 * \f$u(t^{n+1})=u^{n+1}\f$ using the following equations:
 *
 * \f{eqnarray}{
 * v^{(1)} & = & u^n + dt\cdot \mathcal{L}(t^n, u^n)/2,\\
 * v^{(2)} & = & u^n + dt\cdot \mathcal{L}(t^n + dt/2, v^{(1)})/2,\\
 * v^{(3)} & = & u^n + dt\cdot \mathcal{L}(t^n + dt/2, v^{(2)}),\\
 * v^{(4)} & = & u^n + dt\cdot \mathcal{L}(t^n + dt, v^{(3)}),\\
 * u^{n+1} & = & (2v^{(1)} + 4v^{(2)} + 2v^{(3)} + v^{(4)} - 3 u^n)/6.
 * \f}
 *
 * Note that in the implementation, the expression for \f$u^{n+1}\f$ is
 * computed simultaneously with \f$v^{(4)}\f$, so that there are
 * actually only four substeps per step.
 */
class RungeKutta4 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "The standard fourth-order Runge-Kutta time-stepper."};

  RungeKutta4() = default;
  RungeKutta4(const RungeKutta4&) noexcept = default;
  RungeKutta4& operator=(const RungeKutta4&) noexcept = default;
  RungeKutta4(RungeKutta4&&) noexcept = default;
  RungeKutta4& operator=(RungeKutta4&&) noexcept = default;
  ~RungeKutta4() noexcept override = default;

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

  WRAPPED_PUPable_decl_template(RungeKutta4);  // NOLINT

  explicit RungeKutta4(CkMigrateMessage* /*unused*/) noexcept {}

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& p) noexcept override {  // NOLINT
    TimeStepper::Inherit::pup(p);
  }
};

inline bool constexpr operator==(const RungeKutta4& /*lhs*/,
                                 const RungeKutta4& /*rhs*/) noexcept {
  return true;
}

inline bool constexpr operator!=(const RungeKutta4& /*lhs*/,
                                 const RungeKutta4& /*rhs*/) noexcept {
  return false;
}

template <typename Vars, typename DerivVars>
void RungeKutta4::update_u(
    const gsl::not_null<Vars*> u,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(history->integration_order() == 4,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = history->size() - 1;
  const auto& dt_vars = (history->end() - 1).derivative();
  const auto& u0 = history->begin().value();

  switch (substep) {
    case 0: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(1) = u^n + dt * \mathcal{L}(u^n,t^n)/2
      // On entry V = u^n, u0 = u^n, rhs0 = \mathcal{L}(u^n, t^n),
      // time = t^n
      *u += 0.5 * time_step.value() * dt_vars;
      // On exit v = v^(1), time = t^n + (1/2)*dt
      break;
    }
    case 1: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(2) = u^n + dt * \mathcal{L}(v^(1), t^n + (1/2)*dt)/2
      // On entry V = v^(1), u0 = u^n, rhs0 = \mathcal{L}(v^(1), t^n + dt/2),
      // time = t^n + dt
      *u = u0 + 0.5 * time_step.value() * dt_vars;
      // On exit v = v^(2), time = t^n + (1/2)*dt
      break;
    }
    case 2: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(3) = u^n + dt * \mathcal{L}(v^(2), t^n + (1/2)*dt))
      // On entry V = v^(2), u0 = u^n,
      // rhs0 = \mathcal{L}(v^(2), t^n + (1/2)*dt), time = t^n + (1/2)*dt
      *u = u0 + time_step.value() * dt_vars;
      // On exit v = v^(3), time = t^n + dt
      break;
    }
    case 3: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3) + v^(4) - 3*u0)/6
      // On entry V = v^(3), u0 = u^n, rhs0 = \mathcal{L}(v^(3), t^n + dt),
      // time = t^n + dt
      // Note: v^(4) = u0 + dt * \mathcal{L}(t+dt, v^(3)); inserting this gives
      // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3)
      //         + dt*\mathcal{L}(t+dt,v^(3)) - 2*u0)/6
      constexpr double one_sixth = 1.0 / 6.0;
      *u = (2.0 * (history->begin() + 1).value() +
            4.0 * (history->begin() + 2).value() +
            2.0 * (history->begin() + 3).value() +
            (time_step.value() * dt_vars - 2.0 * u0)) *
           one_sixth;
      // On exit v = u^(n+1), time = t^n + dt
      break;
    }
    default:
      ERROR("Substep in RK4 should be one of 0,1,2,3, not " << substep);
  }

  // Clean up old history
  if (history->size() == number_of_substeps()) {
    history->mark_unneeded(history->end());
  }
}

template <typename Vars, typename DerivVars>
bool RungeKutta4::update_u(
    const gsl::not_null<Vars*> u, const gsl::not_null<Vars*> u_error,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(history->integration_order() == 4,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = history->size() - 1;
  if (substep < 3) {
    update_u(u, history, time_step);
  } else {
    const auto& dt_vars = (history->end() - 1).derivative();
    const auto& u0 = history->begin().value();
    switch (substep) {
      case 3: {
        constexpr double prefactor = 1.0 / 32.0;
        *u = (10.0 * (history->begin() + 1).value() +
              14.0 * (history->begin() + 2).value() +
              13.0 * (history->begin() + 3).value() - 5.0 * u0 -
              time_step.value() * dt_vars) *
             prefactor;
        break;
      }
      case 4: {
        // from (17.1.3) of Numerical Recipes 3rd Edition
        // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3) + v^(4) - 3*u0)/6
        // On entry V = v^(3), u0 = u^n, rhs0 = \mathcal{L}(v^(3), t^n + dt),
        // time = t^n + dt
        // Note: v^(4) = u0 + dt * \mathcal{L}(t+dt, v^(3)); inserting this
        // gives u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3)
        //         + dt*\mathcal{L}(t+dt,v^(3)) - 2*u0)/6
        constexpr double one_sixth = 1.0 / 6.0;
        *u = (2.0 * (history->begin() + 1).value() +
              4.0 * (history->begin() + 2).value() +
              2.0 * (history->begin() + 3).value() +
              (time_step.value() * (history->begin() + 3).derivative() -
               2.0 * u0)) *
             one_sixth;
        // On exit v = u^(n+1), time = t^n + dt

        // See Butcher Tableau of Zonneveld 4(3) embedded scheme with five
        // substeps in Table 4.2 of Hairer, Norsett, and Wanner
        *u_error = *u - u0 -
                   time_step.value() *
                       (-3.0 * history->begin().derivative() +
                        14.0 * (history->begin() + 1).derivative() +
                        14.0 * (history->begin() + 2).derivative() +
                        13.0 * (history->begin() + 3).derivative() -
                        32.0 * (history->begin() + 4).derivative()) *
                       one_sixth;
        break;
      }
      default:
        ERROR("Substep in adaptive RK4 should be one of 0,1,2,3,4, not "
              << substep);
    }
  }
  // Clean up old history
  if (history->size() == number_of_substeps_for_error()) {
    history->mark_unneeded(history->end());
  }
  return substep == 4;
}

template <typename Vars, typename DerivVars>
void RungeKutta4::dense_update_u(const gsl::not_null<Vars*> u,
                                 const History<Vars, DerivVars>& history,
                                 const double time) const noexcept {
  ASSERT(history.size() == number_of_substeps(),
         "RK4 can only dense output on last substep ("
             << number_of_substeps() - 1 << "), not substep "
             << history.size() - 1);
  const double step_start = history[0].value();
  const double step_end = history[history.size() - 1].value();
  const double time_step = step_end - step_start;
  const double output_fraction = (time - step_start) / time_step;
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << step_start);
  ASSERT(output_fraction <= 1.0, "Requested time ("
                                     << time << ") not within step ["
                                     << step_start << ", " << step_end << "]");

  // Numerical Recipes Eq. (17.2.15). This implements cubic interpolation
  // throughout the step. Because the history only is available through the
  // penultimate step, i) the value after the step is computed algebraically
  // from previous substeps, and ii) the derivative at the final step is
  // approximated as the derivative at the penultimate substep.
  constexpr double one_sixth = 1.0 / 6.0;
  const auto& u0 = history.begin().value();
  const auto& dt_vars = (history.end() - 1).derivative();
  const Vars u1 =
      (2.0 * (history.begin() + 1).value() +
       4.0 * (history.begin() + 2).value() +
       2.0 * (history.begin() + 3).value() + (time_step * dt_vars - 2.0 * u0)) *
      one_sixth;
  *u = (1.0 - output_fraction) * u0 + output_fraction * u1 +
       (output_fraction) * (output_fraction - 1.0) *
           ((1.0 - 2.0 * output_fraction) * (u1 - u0) +
            (output_fraction - 1.0) * time_step * history.begin().derivative() +
            output_fraction * time_step * dt_vars);
}
}  // namespace TimeSteppers
