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
 * \brief A fifth order RK method constructed with fewer restrictions
 * on its coefficients than is common.  On a standard test suite, it
 * was found to be roughly 10% more efficient than
 * DormandPrince5.\cite Tsitouras2011.
 *
 * The CFL stable step size is 1.7534234969024887.
 */
class Rk5Tsitouras : public RungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "An efficient 5th-order Runge-Kutta time stepper."};

  Rk5Tsitouras() = default;
  Rk5Tsitouras(const Rk5Tsitouras&) = default;
  Rk5Tsitouras& operator=(const Rk5Tsitouras&) = default;
  Rk5Tsitouras(Rk5Tsitouras&&) = default;
  Rk5Tsitouras& operator=(Rk5Tsitouras&&) = default;
  ~Rk5Tsitouras() override = default;

  size_t order() const override;

  size_t error_estimate_order() const override;

  double stable_step() const override;

  WRAPPED_PUPable_decl_template(Rk5Tsitouras);  // NOLINT

  explicit Rk5Tsitouras(CkMigrateMessage* /*msg*/);

  const ButcherTableau& butcher_tableau() const override;
};

inline bool constexpr operator==(const Rk5Tsitouras& /*lhs*/,
                                 const Rk5Tsitouras& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Rk5Tsitouras& /*lhs*/,
                                 const Rk5Tsitouras& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
