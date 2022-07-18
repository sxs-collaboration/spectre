// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class RungeKutta4.

#pragma once

#include <cstddef>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/TimeSteppers/RungeKutta.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace TimeSteppers {
template <typename T>
class UntypedHistory;
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
 *
 * The CFL factor/stable step size is 1.3926467817026411.
 */
class RungeKutta4 : public RungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "The standard fourth-order Runge-Kutta time-stepper."};

  RungeKutta4() = default;
  RungeKutta4(const RungeKutta4&) = default;
  RungeKutta4& operator=(const RungeKutta4&) = default;
  RungeKutta4(RungeKutta4&&) = default;
  RungeKutta4& operator=(RungeKutta4&&) = default;
  ~RungeKutta4() override = default;

  size_t order() const override;

  size_t error_estimate_order() const override;

  double stable_step() const override;

  WRAPPED_PUPable_decl_template(RungeKutta4);  // NOLINT

  explicit RungeKutta4(CkMigrateMessage* /*unused*/) {}

 private:
  const ButcherTableau& butcher_tableau() const override;

  const ButcherTableau& error_tableau() const override;
};

inline bool constexpr operator==(const RungeKutta4& /*lhs*/,
                                 const RungeKutta4& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const RungeKutta4& /*lhs*/,
                                 const RungeKutta4& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
