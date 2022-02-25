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
 * \brief A third order continuous-extension RK method that provides 3rd-order
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
 * in \cite Owren1992 and \cite Gassner20114232. Note that \f$c_1 = 0\f$,
 * \f$s\f$ is the number of stages, and \f$\theta\f$ is the fraction of the
 * step. This is an FSAL stepper.
 *
 * The CFL factor/stable step size is 1.2563726633091645.
 */
class Cerk3 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 3rd-order continuous extension Runge-Kutta method."};

  Cerk3() = default;
  Cerk3(const Cerk3&) = default;
  Cerk3& operator=(const Cerk3&) = default;
  Cerk3(Cerk3&&) = default;
  Cerk3& operator=(Cerk3&&) = default;
  ~Cerk3() override = default;

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

  WRAPPED_PUPable_decl_template(Cerk3);  // NOLINT

  explicit Cerk3(CkMigrateMessage* /*unused*/) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { TimeStepper::Inherit::pup(p); }

 private:
  static constexpr double a2_ = 12.0 / 23.0;
  static constexpr std::array<double, 2> a3_{{-68.0 / 375.0, 368.0 / 375.0}};
  static constexpr std::array<double, 3> a4_{
      {31.0 / 144.0, 529.0 / 1152.0, 125.0 / 384.0}};

  // For the dense output coefficients the index indicates
  // `(degree of theta)`.
  static constexpr std::array<double, 4> b1_{
      {-a4_[0], 1.0, -65.0 / 48.0, 41.0 / 72.0}};
  static constexpr std::array<double, 4> b2_{
      {-a4_[1], 0.0, 529.0 / 384.0, -529.0 / 576.0}};
  static constexpr std::array<double, 4> b3_{
      {-a4_[2], 0.0, 125.0 / 128.0, -125.0 / 192.0}};
  static constexpr std::array<double, 4> b4_{{0.0, 0.0, -1.0, 1.0}};

  // constants for discrete error estimate
  static constexpr std::array<double, 4> e_{
      {1.0 / 24.0, 23.0 / 24.0, 0.0, 0.0}};
  static const std::array<Time::rational_t, 2> c_;
};

inline bool constexpr operator==(const Cerk3& /*lhs*/, const Cerk3& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Cerk3& /*lhs*/, const Cerk3& /*rhs*/) {
  return false;
}

template <typename Vars>
void Cerk3::update_u(const gsl::not_null<Vars*> u,
                     const gsl::not_null<History<Vars>*> history,
                     const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 3,
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
    case 2: {
      *u = u0 + ((a4_[0] - a3_[0]) * dt) * *history->begin().derivative() +
           ((a4_[1] - a3_[1]) * dt) * *(history->begin() + 1).derivative() +
           (a4_[2] * dt) * *(history->begin() + 2).derivative();
      break;
    }
    default:
      ERROR("Bad substep value in Cerk3: " << substep);
  }
}

template <typename Vars, typename ErrVars>
bool Cerk3::update_u(const gsl::not_null<Vars*> u,
                     const gsl::not_null<ErrVars*> u_error,
                     const gsl::not_null<History<Vars>*> history,
                     const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 3,
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  update_u(u, history, time_step);
  const size_t current_substep = (history->end() - 1).time_step_id().substep();
  if (current_substep == 2) {
    const double dt = time_step.value();
    *u_error = ((e_[0] - a4_[0]) * dt) * *history->begin().derivative() +
               ((e_[1] - a4_[1]) * dt) * *(history->begin() + 1).derivative() -
               a4_[2] * dt * *(history->begin() + 2).derivative();
    return true;
  }
  return false;
}

template <typename Vars>
bool Cerk3::dense_update_u(const gsl::not_null<Vars*> u,
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

  // We need the following: k1, k2, k3, k4
  const auto k1 = history.begin().derivative();
  const auto k2 = (history.begin() + 1).derivative();
  const auto k3 = (history.begin() + 2).derivative();
  const auto k4 = (history.begin() + 3).derivative();

  *u = u_n_plus_1 + (dt * evaluate_polynomial(b1_, output_fraction)) * *k1 +
       (dt * evaluate_polynomial(b2_, output_fraction)) * *k2 +
       (dt * evaluate_polynomial(b3_, output_fraction)) * *k3 +
       (dt * evaluate_polynomial(b4_, output_fraction)) * *k4;
  return true;
}
}  // namespace TimeSteppers
