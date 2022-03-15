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
 * \brief A fifth order continuous-extension RK method that provides 5th-order
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
 * The CFL factor/stable step size is 1.5961737362090775.
 */
class Cerk5 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 5th-order continuous extension Runge-Kutta time stepper."};

  Cerk5() = default;
  Cerk5(const Cerk5&) = default;
  Cerk5& operator=(const Cerk5&) = default;
  Cerk5(Cerk5&&) = default;
  Cerk5& operator=(Cerk5&&) = default;
  ~Cerk5() override = default;

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

  WRAPPED_PUPable_decl_template(Cerk5);  // NOLINT

  explicit Cerk5(CkMigrateMessage* /*msg*/);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { TimeStepper::Inherit::pup(p); }

 private:
  static constexpr double a2_ = 1.0 / 6.0;
  static constexpr std::array<double, 2> a3_{{1.0 / 16.0, 3.0 / 16.0}};
  static constexpr std::array<double, 3> a4_{{0.25, -0.75, 1.0}};
  static constexpr std::array<double, 4> a5_{{-0.75, 15.0 / 4.0, -3.0, 0.5}};
  static constexpr std::array<double, 5> a6_{{369.0 / 1372.0, -243.0 / 343.0,
                                              297.0 / 343.0, 1485.0 / 9604.0,
                                              297.0 / 4802.0}};
  static constexpr std::array<double, 6> a7_{
      {-133.0 / 4512.0, 1113.0 / 6016.0, 7945.0 / 16544.0, -12845.0 / 24064.0,
       -315.0 / 24064.0, 156065.0 / 198528.0}};
  // a8 gives the final value and allows us to evaluate k_8 which we need for
  // dense output
  static constexpr std::array<double, 7> a8_{
      {83.0 / 945.0, 0.0, 248.0 / 825.0, 41.0 / 180.0, 1.0 / 36.0,
       2401.0 / 38610.0, 6016.0 / 20475.0}};

  // For the dense output coefficients the index indicates
  // `(degree of theta) - 1`.
  static constexpr std::array<double, 6> b1_{{-a8_[0], 1.0, -3292.0 / 819.0,
                                              17893.0 / 2457.0, -4969.0 / 819.0,
                                              596.0 / 315.0}};
  // b2 = {-a8_[1], 0.0, ...}
  static constexpr double b2_ = -a8_[1];
  static constexpr std::array<double, 6> b3_{{-a8_[2], 0.0, 5112.0 / 715.0,
                                              -43568.0 / 2145.0, 1344.0 / 65.0,
                                              -1984.0 / 275.0}};
  static constexpr std::array<double, 6> b4_{{-a8_[3], 0.0, -123.0 / 52.0,
                                              3161.0 / 234.0, -1465.0 / 78.0,
                                              118.0 / 15.0}};
  static constexpr std::array<double, 6> b5_{
      {-a8_[4], 0.0, -63.0 / 52.0, 1061.0 / 234.0, -413.0 / 78.0, 2.0}};
  static constexpr std::array<double, 6> b6_{
      {-a8_[5], 0.0, -40817.0 / 33462.0, 60025.0 / 50193.0, 2401.0 / 1521.0,
       -9604.0 / 6435.0}};
  static constexpr std::array<double, 6> b7_{
      {-a8_[6], 0.0, 18048.0 / 5915.0, -637696.0 / 53235.0, 96256.0 / 5915.0,
       -48128.0 / 6825.0}};
  static constexpr std::array<double, 6> b8_{
      {0.0, 0.0, -18.0 / 13.0, 75.0 / 13.0, -109.0 / 13.0, 4.0}};

  // constants for discrete error estimate
  static constexpr std::array<double, 8> e_{{-1.0 / 9.0, 0.0, 40.0 / 33.0,
                                             -7.0 / 4.0, -1.0 / 12.0,
                                             343.0 / 198.0, 0.0, 0.0}};
  static const std::array<Time::rational_t, 6> c_;
};

inline bool constexpr operator==(const Cerk5& /*lhs*/, const Cerk5& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Cerk5& /*lhs*/, const Cerk5& /*rhs*/) {
  return false;
}

template <typename Vars>
void Cerk5::update_u(const gsl::not_null<Vars*> u,
                     const gsl::not_null<History<Vars>*> history,
                     const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 5,
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
    case 5: {
      *u = u0 + ((a7_[0] - a6_[0]) * dt) * history->begin().derivative() +
           ((a7_[1] - a6_[1]) * dt) * (history->begin() + 1).derivative() +
           ((a7_[2] - a6_[2]) * dt) * (history->begin() + 2).derivative() +
           ((a7_[3] - a6_[3]) * dt) * (history->begin() + 3).derivative() +
           ((a7_[4] - a6_[4]) * dt) * (history->begin() + 4).derivative() +
           (a7_[5] * dt) * (history->begin() + 5).derivative();
      break;
    }
    case 6: {
      *u = u0 + ((a8_[0] - a7_[0]) * dt) * history->begin().derivative() +
           ((a8_[1] - a7_[1]) * dt) * (history->begin() + 1).derivative() +
           ((a8_[2] - a7_[2]) * dt) * (history->begin() + 2).derivative() +
           ((a8_[3] - a7_[3]) * dt) * (history->begin() + 3).derivative() +
           ((a8_[4] - a7_[4]) * dt) * (history->begin() + 4).derivative() +
           ((a8_[5] - a7_[5]) * dt) * (history->begin() + 5).derivative() +
           (a8_[6] * dt) * (history->begin() + 6).derivative();
      break;
    }
    default:
      ERROR("Bad substep value in CERK5: " << substep);
  }
}

template <typename Vars, typename ErrVars>
bool Cerk5::update_u(const gsl::not_null<Vars*> u,
                     const gsl::not_null<ErrVars*> u_error,
                     const gsl::not_null<History<Vars>*> history,
                     const TimeDelta& time_step) const {
  ASSERT(history->integration_order() == 5,
         "Fixed-order stepper cannot run at order "
             << history->integration_order());
  update_u(u, history, time_step);
  const size_t current_substep = (history->end() - 1).time_step_id().substep();
  if (current_substep == 6) {
    const double dt = time_step.value();
    *u_error = ((e_[0] - a8_[0]) * dt) * history->begin().derivative() +
               ((e_[1] - a8_[1]) * dt) * (history->begin() + 1).derivative() +
               ((e_[2] - a8_[2]) * dt) * (history->begin() + 2).derivative() +
               ((e_[3] - a8_[3]) * dt) * (history->begin() + 3).derivative() +
               ((e_[4] - a8_[4]) * dt) * (history->begin() + 4).derivative() +
               ((e_[5] - a8_[5]) * dt) * (history->begin() + 5).derivative() -
               a8_[6] * dt * (history->begin() + 6).derivative();
    return true;
  }
  return false;
}

template <typename Vars>
bool Cerk5::dense_update_u(const gsl::not_null<Vars*> u,
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

  // We need the following: k1, k2, k3, k4, k5, k6, k7, k8
  const auto& k1 = history.begin().derivative();
  const auto& k2 = (history.begin() + 1).derivative();
  const auto& k3 = (history.begin() + 2).derivative();
  const auto& k4 = (history.begin() + 3).derivative();
  const auto& k5 = (history.begin() + 4).derivative();
  const auto& k6 = (history.begin() + 5).derivative();
  const auto& k7 = (history.begin() + 6).derivative();
  const auto& k8 = (history.begin() + 7).derivative();

  *u = u_n_plus_1 + (dt * evaluate_polynomial(b1_, output_fraction)) * k1 +
       (dt * b2_) * k2 +  //
       (dt * evaluate_polynomial(b3_, output_fraction)) * k3 +
       (dt * evaluate_polynomial(b4_, output_fraction)) * k4 +
       (dt * evaluate_polynomial(b5_, output_fraction)) * k5 +
       (dt * evaluate_polynomial(b6_, output_fraction)) * k6 +
       (dt * evaluate_polynomial(b7_, output_fraction)) * k7 +
       (dt * evaluate_polynomial(b8_, output_fraction)) * k8;
  return true;
}
}  // namespace TimeSteppers
