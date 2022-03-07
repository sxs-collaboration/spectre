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
 * \brief A fourth order continuous-extension RK method that provides 4th-order
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
 * The CFL factor/stable step size is 1.4367588951002057.
 */
class Cerk4 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 4th-order continuous extension Runge-Kutta time stepper."};

  Cerk4() = default;
  Cerk4(const Cerk4&) = default;
  Cerk4& operator=(const Cerk4&) = default;
  Cerk4(Cerk4&&) = default;
  Cerk4& operator=(Cerk4&&) = default;
  ~Cerk4() override = default;

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

  WRAPPED_PUPable_decl_template(Cerk4);  // NOLINT

  explicit Cerk4(CkMigrateMessage* /*msg*/);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { TimeStepper::Inherit::pup(p); }

 private:
  static constexpr double a2_ = 1.0 / 6.0;
  static constexpr std::array<double, 2> a3_{{44.0 / 1369.0, 363.0 / 1369.0}};
  static constexpr std::array<double, 3> a4_{
      {3388.0 / 4913.0, -8349.0 / 4913.0, 8140.0 / 4913.0}};
  static constexpr std::array<double, 4> a5_{
      {-36764.0 / 408375.0, 767.0 / 1125.0, -32708.0 / 136125.0,
       210392.0 / 408375.0}};
  static constexpr std::array<double, 5> a6_{
      {1697.0 / 18876.0, 0.0, 50653.0 / 116160.0, 299693.0 / 1626240.0,
       3375.0 / 11648.0}};

  // For the dense output coefficients the index indicates
  // `(degree of theta) - 1`.
  static constexpr std::array<double, 5> b1_{{-a6_[0], 1.0, -104217.0 / 37466.0,
                                              1806901.0 / 618189.0,
                                              -866577.0 / 824252.0}};
  // b2 = {-a8_[1], 0.0, ...}
  static constexpr double b2_ = -a6_[1];
  static constexpr std::array<double, 5> b3_{{-a6_[2], 0.0, 861101.0 / 230560.0,
                                              -2178079.0 / 380424.0,
                                              12308679.0 / 5072320.0}};
  static constexpr std::array<double, 5> b4_{{-a6_[3], 0.0, -63869.0 / 293440.0,
                                              6244423.0 / 5325936.0,
                                              -7816583.0 / 10144640.0}};
  static constexpr std::array<double, 5> b5_{
      {-a6_[4], 0.0, -1522125.0 / 762944.0, 982125.0 / 190736.0,
       -624375.0 / 217984.0}};
  static constexpr std::array<double, 5> b6_{
      {0.0, 0.0, 165.0 / 131.0, -461.0 / 131.0, 296.0 / 131.0}};

  // constants for discrete error estimate
  static constexpr std::array<double, 6> e_{
      {101.0 / 363.0, 0.0, -1369.0 / 14520.0, 11849.0 / 14520.0, 0.0, 0.0}};
  static const std::array<Time::rational_t, 4> c_;
};

inline bool constexpr operator==(const Cerk4& /*lhs*/, const Cerk4& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Cerk4& /*lhs*/, const Cerk4& /*rhs*/) {
  return false;
}

template <typename Vars>
void Cerk4::update_u(const gsl::not_null<Vars*> u,
                     const gsl::not_null<History<Vars>*> history,
                     const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 4,
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
      *u = u0 + (a2_ * dt) * history->begin().derivative();
      break;
    }
    case 1: {
      *u = u0 + ((a3_[0] - a2_) * dt) * history->begin().derivative() +
           (a3_[1] * dt) * (history->begin() + 1).derivative();
      break;
    }
    case 2: {
      *u = u0 + ((a4_[0] - a3_[0]) * dt) * history->begin().derivative() +
           ((a4_[1] - a3_[1]) * dt) * (history->begin() + 1).derivative() +
           (a4_[2] * dt) * (history->begin() + 2).derivative();
      break;
    }
    case 3: {
      *u = u0 + ((a5_[0] - a4_[0]) * dt) * history->begin().derivative() +
           ((a5_[1] - a4_[1]) * dt) * (history->begin() + 1).derivative() +
           ((a5_[2] - a4_[2]) * dt) * (history->begin() + 2).derivative() +
           (a5_[3] * dt) * (history->begin() + 3).derivative();
      break;
    }
    case 4: {
      *u = u0 + ((a6_[0] - a5_[0]) * dt) * history->begin().derivative() +
           ((a6_[1] - a5_[1]) * dt) * (history->begin() + 1).derivative() +
           ((a6_[2] - a5_[2]) * dt) * (history->begin() + 2).derivative() +
           ((a6_[3] - a5_[3]) * dt) * (history->begin() + 3).derivative() +
           (a6_[4] * dt) * (history->begin() + 4).derivative();
      break;
    }
    default:
      ERROR("Bad substep value in CERK4: " << substep);
  }
}

template <typename Vars, typename ErrVars>
bool Cerk4::update_u(const gsl::not_null<Vars*> u,
                     const gsl::not_null<ErrVars*> u_error,
                     const gsl::not_null<History<Vars>*> history,
                     const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 4,
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  update_u(u, history, time_step);
  const size_t current_substep = (history->end() - 1).time_step_id().substep();
  if (current_substep == 4) {
    const double dt = time_step.value();
    *u_error = ((e_[0] - a6_[0]) * dt) * history->begin().derivative() +
               ((e_[1] - a6_[1]) * dt) * (history->begin() + 1).derivative() +
               ((e_[2] - a6_[2]) * dt) * (history->begin() + 2).derivative() +
               ((e_[3] - a6_[3]) * dt) * (history->begin() + 3).derivative() -
               (a6_[4] * dt) * (history->begin() + 4).derivative();
    return true;
  }
  return false;
}

template <typename Vars>
bool Cerk4::dense_update_u(const gsl::not_null<Vars*> u,
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

  // We need the following: k1, k2, k3, k4, k5, k6
  const auto& k1 = history.begin().derivative();
  const auto& k2 = (history.begin() + 1).derivative();
  const auto& k3 = (history.begin() + 2).derivative();
  const auto& k4 = (history.begin() + 3).derivative();
  const auto& k5 = (history.begin() + 4).derivative();
  const auto& k6 = (history.begin() + 5).derivative();

  *u = u_n_plus_1 + (dt * evaluate_polynomial(b1_, output_fraction)) * k1 +
       (dt * b2_) * k2 +  //
       (dt * evaluate_polynomial(b3_, output_fraction)) * k3 +
       (dt * evaluate_polynomial(b4_, output_fraction)) * k4 +
       (dt * evaluate_polynomial(b5_, output_fraction)) * k5 +
       (dt * evaluate_polynomial(b6_, output_fraction)) * k6;
  return true;
}
}  // namespace TimeSteppers
