// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DormandPrince5.

#pragma once

#include <cstddef>

#include "Options/String.hpp"
#include "Time/TimeSteppers/RungeKutta.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

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
class DormandPrince5 : public RungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "The standard Dormand-Prince 5th-order time stepper."};

  DormandPrince5() = default;
  DormandPrince5(const DormandPrince5&) = default;
  DormandPrince5& operator=(const DormandPrince5&) = default;
  DormandPrince5(DormandPrince5&&) = default;
  DormandPrince5& operator=(DormandPrince5&&) = default;
  ~DormandPrince5() override = default;

  size_t order() const override;

  size_t error_estimate_order() const override;

  double stable_step() const override;

  WRAPPED_PUPable_decl_template(DormandPrince5);  // NOLINT

  explicit DormandPrince5(CkMigrateMessage* /*unused*/) {}

  const ButcherTableau& butcher_tableau() const override;
};

inline bool constexpr operator==(const DormandPrince5& /*lhs*/,
                                 const DormandPrince5& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const DormandPrince5& /*lhs*/,
                                 const DormandPrince5& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
