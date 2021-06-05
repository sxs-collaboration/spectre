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
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  switch (substep) {
    case 0: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(1) = u^n + dt * \mathcal{L}(u^n,t^n)/2
      *u = (history->end() - 1).value() +
           0.5 * time_step.value() * history->begin().derivative();
      break;
    }
    case 1: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(2) = u^n + dt * \mathcal{L}(v^(1), t^n + (1/2)*dt)/2
      *u = (history->end() - 1).value() -
           0.5 * time_step.value() *
               (history->begin().derivative() -
                (history->begin() + 1).derivative());
      break;
    }
    case 2: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // v^(3) = u^n + dt * \mathcal{L}(v^(2), t^n + (1/2)*dt))
      *u = (history->end() - 1).value() +
           time_step.value() * (-0.5 * (history->begin() + 1).derivative() +
                                (history->begin() + 2).derivative());
      break;
    }
    case 3: {
      // from (17.1.3) of Numerical Recipes 3rd Edition
      // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3) + v^(4) - 3*u0)/6
      // Note: v^(4) = u0 + dt * \mathcal{L}(t+dt, v^(3)); inserting this gives
      // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3)
      //         + dt*\mathcal{L}(t+dt,v^(3)) - 2*u0)/6
      *u = (history->end() - 1).value() +
           (1.0 / 3.0) * time_step.value() *
               (0.5 * history->begin().derivative() +
                (history->begin() + 1).derivative() -
                2.0 * (history->begin() + 2).derivative() +
                0.5 * (history->begin() + 3).derivative());
      break;
    }
    default:
      ERROR("Substep in RK4 should be one of 0,1,2,3, not " << substep);
  }
}

template <typename Vars, typename ErrVars, typename DerivVars>
bool RungeKutta4::update_u(
    const gsl::not_null<Vars*> u, const gsl::not_null<ErrVars*> u_error,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(history->integration_order() == 4,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();
  if (substep < 3) {
    update_u(u, history, time_step);
  } else {
    switch (substep) {
      case 3: {
        *u = (history->end() - 1).value() +
             (1.0 / 32.0) * time_step.value() *
                 (5.0 * history->begin().derivative() +
                  7.0 * (history->begin() + 1).derivative() -
                  19.0 * (history->begin() + 2).derivative() -
                  (history->begin() + 3).derivative());
        break;
      }
      case 4: {
        // from (17.1.3) of Numerical Recipes 3rd Edition
        // u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3) + v^(4) - 3*u0)/6
        // Note: v^(4) = u0 + dt * \mathcal{L}(t+dt, v^(3)); inserting this
        // gives u^(n+1) = (2v^(1) + 4*v^(2) + 2*v^(3)
        //         + dt*\mathcal{L}(t+dt,v^(3)) - 2*u0)/6
        *u = (history->end() - 1).value() +
             (1.0 / 96.0) * time_step.value() *
                 (history->begin().derivative() +
                  11.0 * (history->begin() + 1).derivative() -
                  7.0 * (history->begin() + 2).derivative() +
                  19.0 * (history->begin() + 3).derivative());

        // See Butcher Tableau of Zonneveld 4(3) embedded scheme with five
        // substeps in Table 4.2 of Hairer, Norsett, and Wanner
        *u_error = (2.0 / 3.0) * time_step.value() *
                   (history->begin().derivative() -
                    3.0 * ((history->begin() + 1).derivative() +
                           (history->begin() + 2).derivative() +
                           (history->begin() + 3).derivative()) +
                    8.0 * (history->begin() + 4).derivative());
        break;
      }
      default:
        ERROR("Substep in adaptive RK4 should be one of 0,1,2,3,4, not "
              << substep);
    }
  }
  return substep == 4;
}

template <typename Vars, typename DerivVars>
bool RungeKutta4::dense_update_u(const gsl::not_null<Vars*> u,
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
    *u = (history.end() - 1).value();
    return true;
  }
  const evolution_less<double> before{step_end > step_start};
  if (history.size() == 1 or before(step_end, time)) {
    return false;
  }
  const double time_step = step_end - step_start;
  const double output_fraction = (time - step_start) / time_step;
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << step_start);
  ASSERT(output_fraction <= 1.0, "Requested time ("
                                     << time << ") not within step ["
                                     << step_start << ", " << step_end << "]");

  // Numerical Recipes Eq. (17.2.15). This implements cubic interpolation
  // throughout the step.
  *u = (history.end() - 1).value() -
       (1.0 / 3.0) * time_step * (1.0 - output_fraction) *
           ((1.0 - output_fraction) *
                ((0.5 - 2.0 * output_fraction) * history.begin().derivative() +
                 (1.0 + 2.0 * output_fraction) *
                     ((history.begin() + 1).derivative() +
                      (history.begin() + 2).derivative() +
                      0.5 * (history.begin() + 3).derivative())) +
            3.0 * square(output_fraction) * (history.end() - 1).derivative());
  return true;
}
}  // namespace TimeSteppers
