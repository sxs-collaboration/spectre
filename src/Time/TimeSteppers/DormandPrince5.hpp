// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DormandPrince5.

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <pup.h>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ConstantExpressions.hpp"
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

  template <typename Vars, typename DerivVars>
  void dense_update_u(gsl::not_null<Vars*> u,
                      const History<Vars, DerivVars>& history,
                      double time) const noexcept;

  uint64_t number_of_substeps() const noexcept override;

  size_t number_of_past_steps() const noexcept override;

  double stable_step() const noexcept override;

  TimeStepId next_time_id(const TimeStepId& current_id,
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
  static constexpr double _a2 = 0.2;
  static constexpr std::array<double, 2> _a3{{3.0 / 40.0, 9.0 / 40.0}};
  static constexpr std::array<double, 3> _a4{
      {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0}};
  static constexpr std::array<double, 4> _a5{
      {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0}};
  static constexpr std::array<double, 5> _a6{{9017.0 / 3168.0, -355.0 / 33.0,
                                              46732.0 / 5247.0, 49.0 / 176.0,
                                              -5103.0 / 18656.0}};
  static constexpr std::array<double, 6> _b{{35.0 / 384.0, 0.0, 500.0 / 1113.0,
                                             125.0 / 192.0, -2187.0 / 6784.0,
                                             11.0 / 84.0}};
  static const std::array<Time::rational_t, 5> _c;

  // Coefficients for dense output, taken from Sec. 7.2 of
  // \cite NumericalRecipes
  static constexpr std::array<double, 6> _d{
      {-12715105075.0 / 11282082432.0, 87487479700.0 / 32700410799.0,
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
  const size_t substep = history->size() - 1;
  const auto& u0 = history->begin().value();
  const double dt = time_step.value();

  const auto increment_u = [&u, &history, &dt](const auto& coeffs) noexcept {
    for (size_t i = 0; i < coeffs.size(); ++i) {
      *u += (gsl::at(coeffs, i) * dt) *
            (history->begin() + static_cast<int>(i)).derivative();
    }
  };

  if (substep == 0) {
    *u += (_a2 * dt) * history->begin().derivative();
  } else if (substep < 6) {
    *u = u0;
    if (substep == 1) {
      increment_u(_a3);
    } else if (substep == 2) {
      increment_u(_a4);
    } else if (substep == 3) {
      increment_u(_a5);
    } else if (substep == 4) {
      increment_u(_a6);
    } else {
      increment_u(_b);
    }
  } else {
    ERROR("Substep in DP5 should be one of 0,1,2,3,4,5, not " << substep);
  }

  // Clean up old history
  if (history->size() == number_of_substeps()) {
    history->mark_unneeded(history->end());
  }
}

template <typename Vars, typename DerivVars>
void DormandPrince5::dense_update_u(const gsl::not_null<Vars*> u,
                                    const History<Vars, DerivVars>& history,
                                    const double time) const noexcept {
  ASSERT(history.size() == number_of_substeps(),
         "DP5 can only dense output on last substep ("
             << number_of_substeps() - 1 << "), not substep "
             << history.size() - 1);
  const double t0 = history[0].value();
  const double t_end = history[history.size() - 1].value();
  // The history does not contain the final step; specifically,
  // step_end = t0 + c[4] * dt, so dt = (t_end - t0) / c[4]. But since
  // c[4] = 1.0 for DP5, we don't need to divide by c[4] here.
  const double dt = t_end - t0;
  const double output_fraction = (time - t0) / dt;
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << t0);
  ASSERT(output_fraction <= 1.0, "Requested time ("
                                     << time << ") not within step [" << t0
                                     << ", " << t0 + dt << "]");

  // Get the solution u1 at the final time
  const auto& u0 = history.begin().value();
  Vars u1 = u0;
  for (size_t i = 0; i < 6; ++i) {
    u1 += (gsl::at(_b, i) * dt) *
          (history.begin() + static_cast<int>(i)).derivative();
  }

  // We need the following: k1, k3, k4, k5, k6.
  // Here k1 = dt * l1, k3 = dt * l3, etc.
  const auto& l1 = history.begin().derivative();
  const auto& l3 = (history.begin() + 2).derivative();
  const auto& l4 = (history.begin() + 3).derivative();
  const auto& l5 = (history.begin() + 4).derivative();
  const auto& l6 = (history.begin() + 5).derivative();

  // Compute the updating coefficents, called rcontN in Numerical recipes,
  // that will be reused, so I don't have to compute them more than once.
  const Vars rcont2 = u1 - u0;
  const Vars rcont3 = dt * l1 - rcont2;

  // The formula for dense output is given in Numerical Recipes Sec. 17.2.3.
  // Note: L(t+dt, u1) after the step is unavailable in the history; so here,
  // approximate L(t+dt, u1) by l6.
  *u = u0 + output_fraction *
                (rcont2 +
                 (1.0 - output_fraction) *
                     (rcont3 + output_fraction *
                                   ((rcont2 - dt * l6 - rcont3) +
                                    ((1.0 - output_fraction) * dt) *
                                        (_d[0] * l1 + _d[1] * l3 + _d[2] * l4 +
                                         _d[3] * l5 + (_d[4] + _d[5]) * l6))));
}
}  // namespace TimeSteppers
