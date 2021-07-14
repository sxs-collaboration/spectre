// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DormandPrince5.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <pup.h>
#include <tuple>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
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
 * The standard 5th-order Dormand-Prince time stepping method, given e.g. in
 * Sec. 7.2 of \cite NumericalRecipes.
 *
 * \f{eqnarray}{
 * \frac{du}{dt} & = & \mathcal{L}(t,u).
 * \f}
 * Given a solution \f$u(t^n)=u^n\f$, this stepper computes
 * \f$u(t^{n+1})=u^{n+1}\f$ using the following equations:
 *
 * \f{align}{
 * k^{(1)} & = dt \mathcal{L}(t^n, u^n),\\
 * k^{(i)} & = dt \mathcal{L}(t^n + c_i dt,
 *                              u^n + \sum_{j=1}^{i-1} a_{ij} k^{(j)}),
 *                              \mbox{ } 2 \leq i \leq 6,\\
 * u^{n+1} & = u^n + \sum_{i=1}^{6} b_i k^{(i)}.
 * \f}
 *
 * Here the coefficients \f$a_{ij}\f$, \f$b_i\f$, and \f$c_i\f$ are given
 * in e.g. Sec. 7.2 of \cite NumericalRecipes. Note that \f$c_1 = 0\f$.
 *
 * The CFL factor/stable step size is 1.6532839463174733.
 */
class DormandPrince5 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "The standard Dormand-Prince 5th-order time stepper."};

  DormandPrince5() = default;
  DormandPrince5(const DormandPrince5&) noexcept = default;
  DormandPrince5& operator=(const DormandPrince5&) noexcept = default;
  DormandPrince5(DormandPrince5&&) noexcept = default;
  DormandPrince5& operator=(DormandPrince5&&) noexcept = default;
  ~DormandPrince5() noexcept override = default;

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

  WRAPPED_PUPable_decl_template(DormandPrince5);  // NOLINT

  explicit DormandPrince5(CkMigrateMessage* /*unused*/) noexcept {}

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& p) noexcept override {  // NOLINT
    TimeStepper::Inherit::pup(p);
  }

 private:
  // Coefficients from the Dormand-Prince 5 Butcher tableau (e.g. Sec. 7.2
  // of \cite NumericalRecipes).
  static constexpr std::array<double, 1> a2_{{0.2}};
  static constexpr std::array<double, 2> a3_{{3.0 / 40.0, 9.0 / 40.0}};
  static constexpr std::array<double, 3> a4_{
      {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0}};
  static constexpr std::array<double, 4> a5_{
      {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0}};
  static constexpr std::array<double, 5> a6_{{9017.0 / 3168.0, -355.0 / 33.0,
                                              46732.0 / 5247.0, 49.0 / 176.0,
                                              -5103.0 / 18656.0}};
  static constexpr std::array<double, 6> b_{{35.0 / 384.0, 0.0, 500.0 / 1113.0,
                                             125.0 / 192.0, -2187.0 / 6784.0,
                                             11.0 / 84.0}};
  // Coefficients for the embedded method, for generating an error measure
  // during adaptive stepping (e.g. Sec. 7.2 of \cite NumericalRecipes).
  static constexpr std::array<double, 7> b_alt_{
      {5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
       -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0}};

  static const std::array<Time::rational_t, 6> c_;

  // Coefficients for dense output, taken from Sec. 7.2 of
  // \cite NumericalRecipes
  static constexpr std::array<double, 7> d_{
      {-12715105075.0 / 11282082432.0, 0.0, 87487479700.0 / 32700410799.0,
       -10690763975.0 / 1880347072.0, 701980252875.0 / 199316789632.0,
       -1453857185.0 / 822651844.0, 69997945.0 / 29380423.0}};
};

inline bool constexpr operator==(const DormandPrince5& /*lhs*/,
                                 const DormandPrince5& /*rhs*/) noexcept {
  return true;
}

inline bool constexpr operator!=(const DormandPrince5& /*lhs*/,
                                 const DormandPrince5& /*rhs*/) noexcept {
  return false;
}

template <typename Vars, typename DerivVars>
void DormandPrince5::update_u(
    const gsl::not_null<Vars*> u,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(history->integration_order() == 5,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  // Clean up old history
  if (substep == 0) {
    history->mark_unneeded(history->end() - 1);
  }

  const double dt = time_step.value();

  const auto increment_u = [&u, &history, &dt](
                               const auto& coeffs_last,
                               const auto& coeffs_this) noexcept {
    static_assert(std::tuple_size_v<std::decay_t<decltype(coeffs_last)>> + 1 ==
                      std::tuple_size_v<std::decay_t<decltype(coeffs_this)>>,
                  "Unexpected coefficient vector sizes.");
    *u = history->most_recent_value() +
         coeffs_this.back() * dt * (history->end() - 1).derivative();
    for (size_t i = 0; i < coeffs_last.size(); ++i) {
      *u += (gsl::at(coeffs_this, i) - gsl::at(coeffs_last, i)) * dt *
            (history->begin() + static_cast<int>(i)).derivative();
    }
  };

  if (substep == 0) {
    *u = history->most_recent_value() +
         (a2_[0] * dt) * history->begin().derivative();
  } else if (substep == 1) {
    increment_u(a2_, a3_);
  } else if (substep == 2) {
    increment_u(a3_, a4_);
  } else if (substep == 3) {
    increment_u(a4_, a5_);
  } else if (substep == 4) {
    increment_u(a5_, a6_);
  } else if (substep == 5) {
    increment_u(a6_, b_);
  } else {
    ERROR("Substep in DP5 should be one of 0,1,2,3,4,5, not " << substep);
  }
}

template <typename Vars, typename ErrVars, typename DerivVars>
bool DormandPrince5::update_u(
    const gsl::not_null<Vars*> u, const gsl::not_null<ErrVars*> u_error,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(history->integration_order() == 5,
         "Fixed-order stepper cannot run at order "
         << history->integration_order());
  const size_t substep = (history->end() - 1).time_step_id().substep();

  if (substep < 6) {
    update_u(u, history, time_step);
  } else if (substep == 6) {
    // u is the same as for the previous substep.
    *u = history->most_recent_value();

    const double dt = time_step.value();

    *u_error = -b_alt_.back() * dt * (history->end() - 1).derivative();
    for (size_t i = 0; i < b_.size(); ++i) {
      *u_error -= (gsl::at(b_alt_, i) - gsl::at(b_, i)) * dt *
                  (history->begin() + static_cast<int>(i)).derivative();
    }
  } else {
    ERROR("Substep in adaptive DP5 should be one of 0,1,2,3,4,5,6, not "
          << substep);
  }
  return substep == 6;
}

template <typename Vars, typename DerivVars>
bool DormandPrince5::dense_update_u(const gsl::not_null<Vars*> u,
                                    const History<Vars, DerivVars>& history,
                                    const double time) const noexcept {
  if ((history.end() - 1).time_step_id().substep() != 0) {
    return false;
  }
  const double t0 = history.front().value();
  const double t_end = history.back().value();
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

  // The formula for dense output is given in Numerical Recipes Sec. 17.2.3.
  // This version is modified to eliminate all the values of the function
  // except the most recent.
  const auto common = [&output_fraction](const size_t n) noexcept {
    return square(output_fraction) * gsl::at(d_, n) -
           (1.0 + 2.0 * output_fraction) * gsl::at(b_, n);
  };
  *u = history.most_recent_value() +
       dt * (1.0 - output_fraction) *
           ((1.0 - output_fraction) *
                ((common(0) + output_fraction) * history.begin().derivative() +
                 common(2) * (history.begin() + 2).derivative() +
                 common(3) * (history.begin() + 3).derivative() +
                 common(4) * (history.begin() + 4).derivative() +
                 common(5) * (history.begin() + 5).derivative()) +
            square(output_fraction) * ((1.0 - output_fraction) * d_[6] - 1.0) *
                (history.begin() + 6).derivative());
  return true;
}
}  // namespace TimeSteppers
