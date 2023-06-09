// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Options/String.hpp"
#include "Time/TimeSteppers/RungeKutta.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

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
class Rk4Owren : public RungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 4th-order continuous extension Runge-Kutta time stepper."};

  Rk4Owren() = default;
  Rk4Owren(const Rk4Owren&) = default;
  Rk4Owren& operator=(const Rk4Owren&) = default;
  Rk4Owren(Rk4Owren&&) = default;
  Rk4Owren& operator=(Rk4Owren&&) = default;
  ~Rk4Owren() override = default;

  size_t order() const override;

  size_t error_estimate_order() const override;

  double stable_step() const override;

  WRAPPED_PUPable_decl_template(Rk4Owren);  // NOLINT

  explicit Rk4Owren(CkMigrateMessage* /*msg*/);

  const ButcherTableau& butcher_tableau() const override;
};

inline bool constexpr operator==(const Rk4Owren& /*lhs*/,
                                 const Rk4Owren& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Rk4Owren& /*lhs*/,
                                 const Rk4Owren& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
