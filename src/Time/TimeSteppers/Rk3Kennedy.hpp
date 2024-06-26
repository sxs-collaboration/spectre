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
 * \brief A third-order Runge-Kutta method with IMEX support.
 *
 * The coefficients are given as ARK3(2)4L[2]SA in \cite Kennedy2003.
 *
 * The implicit part is stiffly accurate and L-stable.
 *
 * The CFL factor/stable step size is 1.832102281377816.
 */
class Rk3Kennedy : public ImexRungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 3rd-order Runge-Kutta scheme devised by Kennedy and Carpenter."};

  Rk3Kennedy() = default;
  Rk3Kennedy(const Rk3Kennedy&) = default;
  Rk3Kennedy& operator=(const Rk3Kennedy&) = default;
  Rk3Kennedy(Rk3Kennedy&&) = default;
  Rk3Kennedy& operator=(Rk3Kennedy&&) = default;
  ~Rk3Kennedy() override = default;

  size_t order() const override;

  double stable_step() const override;

  size_t imex_order() const override;

  size_t implicit_stage_order() const override;

  WRAPPED_PUPable_decl_template(Rk3Kennedy);  // NOLINT

  explicit Rk3Kennedy(CkMigrateMessage* /*unused*/) {}

  const ButcherTableau& butcher_tableau() const override;

  const ImplicitButcherTableau& implicit_butcher_tableau() const override;
};

inline bool constexpr operator==(const Rk3Kennedy& /*lhs*/,
                                 const Rk3Kennedy& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Rk3Kennedy& lhs, const Rk3Kennedy& rhs) {
  return not(lhs == rhs);
}
}  // namespace TimeSteppers
