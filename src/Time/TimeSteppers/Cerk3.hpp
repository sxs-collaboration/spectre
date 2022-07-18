// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/TimeSteppers/RungeKutta.hpp"
#include "Utilities/TMPL.hpp"

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
class Cerk3 : public RungeKutta {
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

  size_t order() const override;

  size_t error_estimate_order() const override;

  double stable_step() const override;

  WRAPPED_PUPable_decl_template(Cerk3);  // NOLINT

  explicit Cerk3(CkMigrateMessage* /*unused*/) {}

 private:
  const ButcherTableau& butcher_tableau() const override;
};

inline bool constexpr operator==(const Cerk3& /*lhs*/, const Cerk3& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Cerk3& /*lhs*/, const Cerk3& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
