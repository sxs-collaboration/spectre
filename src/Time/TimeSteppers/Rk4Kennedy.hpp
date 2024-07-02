// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Options/String.hpp"
#include "Time/TimeSteppers/ImexRungeKutta.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {
/*!
 * \ingroup TimeSteppersGroup
 * \brief A fourth-order Runge-Kutta method with IMEX support.
 *
 * The coefficients are given as ARK4(3)6L[2]SA in \cite Kennedy2003.
 *
 * The implicit part is stiffly accurate and L-stable.
 *
 * The CFL factor/stable step size is 2.1172491998184686.
 */
class Rk4Kennedy : public ImexRungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 4th-order Runge-Kutta scheme devised by Kennedy and Carpenter."};

  Rk4Kennedy() = default;
  Rk4Kennedy(const Rk4Kennedy&) = default;
  Rk4Kennedy& operator=(const Rk4Kennedy&) = default;
  Rk4Kennedy(Rk4Kennedy&&) = default;
  Rk4Kennedy& operator=(Rk4Kennedy&&) = default;
  ~Rk4Kennedy() override = default;

  size_t order() const override;

  double stable_step() const override;

  size_t imex_order() const override;

  size_t implicit_stage_order() const override;

  WRAPPED_PUPable_decl_template(Rk4Kennedy);  // NOLINT

  explicit Rk4Kennedy(CkMigrateMessage* /*unused*/) {}

  const ButcherTableau& butcher_tableau() const override;

  const ImplicitButcherTableau& implicit_butcher_tableau() const override;
};

inline bool constexpr operator==(const Rk4Kennedy& /*lhs*/,
                                 const Rk4Kennedy& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Rk4Kennedy& lhs, const Rk4Kennedy& rhs) {
  return not(lhs == rhs);
}
}  // namespace TimeSteppers
