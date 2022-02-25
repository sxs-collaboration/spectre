// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct TimeStepId;
namespace TimeSteppers {
template <typename Vars>
class History;
}  // namespace TimeSteppers
/// \endcond

namespace TimeSteppers {
/*!
 * \ingroup TimeSteppersGroup
 * \brief A second order continuous-extension RK method that provides 2nd-order
 * dense output.
 *
 * \f{eqnarray}{
 * \frac{du}{dt} & = & \mathcal{L}(t,u).
 * \f}
 * Given a solution \f$u(t^n)=u^n\f$, this stepper computes
 * \f$u(t^{n+1})=u^{n+1}\f$ using the following equations:
 *
 * \f{align}{
 * k^{(i)} & = \mathcal{L}(t^n + c_i \Delta t,
 *                         u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} k^{(j)}),
 *                              \mbox{ } 1 \leq i \leq s,\\
 * u^{n+1}(t^n + \theta \Delta t) & = u^n + \Delta t \sum_{i=1}^{s} b_i(\theta)
 * k^{(i)}. \f}
 *
 * Here the coefficients \f$a_{ij}\f$, \f$b_i\f$, and \f$c_i\f$ are given
 * in \cite Gassner20114232. Note that \f$c_1 = 0\f$, \f$s\f$ is the number
 * of stages, and \f$\theta\f$ is the fraction of the step.
 *
 * The CFL factor/stable step size is 1.0.
 */
class Cerk2 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 2nd order accurate continuous extension Runge-Kutta method.."};

  Cerk2() = default;
  Cerk2(const Cerk2&) = default;
  Cerk2& operator=(const Cerk2&) = default;
  Cerk2(Cerk2&&) = default;
  Cerk2& operator=(Cerk2&&) = default;
  ~Cerk2() override = default;

  template <typename Vars>
  void update_u(gsl::not_null<Vars*> u, gsl::not_null<History<Vars>*> history,
                const TimeDelta& time_step) const;

  template <typename Vars, typename ErrVars>
  bool update_u(gsl::not_null<Vars*> u, gsl::not_null<ErrVars*> u_error,
                gsl::not_null<History<Vars>*> history,
                const TimeDelta& time_step) const;

  template <typename Vars>
  bool dense_update_u(gsl::not_null<Vars*> u, const History<Vars>& history,
                      double time) const;

  size_t order() const override;

  size_t error_estimate_order() const override;

  uint64_t number_of_substeps() const override;

  uint64_t number_of_substeps_for_error() const override;

  size_t number_of_past_steps() const override;

  double stable_step() const override;

  TimeStepId next_time_id(const TimeStepId& current_id,
                          const TimeDelta& time_step) const override;

  TimeStepId next_time_id_for_error(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const override;

  template <typename Vars>
  bool can_change_step_size(
      const TimeStepId& time_id,
      const TimeSteppers::History<Vars>& /*history*/) const {
    return time_id.substep() == 0;
  }

  WRAPPED_PUPable_decl_template(Cerk2);  // NOLINT

  explicit Cerk2(CkMigrateMessage* /*unused*/) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { TimeStepper::Inherit::pup(p); }

 private:
  static constexpr double a2_ = 1.0;
  static constexpr std::array<double, 2> a3_{{0.5, 0.5}};

  // For the dense output coefficients the index indicates
  // `(degree of theta)`.
  static constexpr std::array<double, 3> b1_{{-a3_[0], 1.0, -0.5}};
  static constexpr std::array<double, 3> b2_{{-a3_[1], 0.0, 0.5}};

  // constants for discrete error estimate
  static constexpr std::array<double, 2> e_{{1.0, 0.0}};
  static const std::array<Time::rational_t, 1> c_;
};

inline bool constexpr operator==(const Cerk2& /*lhs*/, const Cerk2& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Cerk2& /*lhs*/, const Cerk2& /*rhs*/) {
  return false;
}

template <typename Vars>
void Cerk2::update_u(const gsl::not_null<Vars*> u,
                     const gsl::not_null<History<Vars>*> history,
                     const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 2,
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  const auto& u0 = history->most_recent_value();
  const double dt = time_step.value();

  switch (substep) {
    case 0: {
      *u = u0 + (a2_ * dt) * *history->begin().derivative();
      break;
    }
    case 1: {
      *u = u0 + ((a3_[0] - a2_) * dt) * *history->begin().derivative() +
           (a3_[1] * dt) * *(history->begin() + 1).derivative();
      break;
    }
    default:
      ERROR("Bad substep value in Cerk2: " << substep);
  }
}

template <typename Vars, typename ErrVars>
bool Cerk2::update_u(const gsl::not_null<Vars*> u,
                     const gsl::not_null<ErrVars*> u_error,
                     const gsl::not_null<History<Vars>*> history,
                     const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 2,
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  update_u(u, history, time_step);
  const size_t current_substep = (history->end() - 1).time_step_id().substep();
  if (current_substep == 1) {
    const double dt = time_step.value();
    *u_error = ((e_[0] - a3_[0]) * dt) * *history->begin().derivative() -
               a3_[1] * dt * *(history->begin() + 1).derivative();
    return true;
  }
  return false;
}

template <typename Vars>
bool Cerk2::dense_update_u(const gsl::not_null<Vars*> u,
                           const History<Vars>& history,
                           const double time) const {
  if ((history.end() - 1).time_step_id().substep() != 0) {
    return false;
  }
  const double t0 = history[0].value();
  const double t_end = history[history.size() - 1].value();
  if (time == t_end) {
    // Special case necessary for dense output at the initial time,
    // before taking a step.
    *u = history.most_recent_value();
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

  const auto& u_n_plus_1 = history.most_recent_value();

  // We need the following: k1, k2
  const auto k1 = history.begin().derivative();
  const auto k2 = (history.begin() + 1).derivative();

  *u = u_n_plus_1 + (dt * evaluate_polynomial(b1_, output_fraction)) * *k1 +
       (dt * evaluate_polynomial(b2_, output_fraction)) * *k2;
  return true;
}
}  // namespace TimeSteppers
