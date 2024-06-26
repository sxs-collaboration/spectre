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
 * The method as published has four stages, but is implemented with
 * five as a way to convert it to an EDIRK method.
 *
 * The coefficients are given as IMEX-SSP3(4,3,3) in \cite Pareschi2005.
 *
 * While this method can be implemented so that the explicit part is
 * strong-stability-preserving, the presentation in \cite Pareschi2005
 * is not, and this implementation follows that presentation.  See
 * \cite HesthavenWarburton section 5.7 for details.
 *
 * Using this time stepper in a non-IMEX simulation is not
 * recommended, as it performs two unused RHS evaluations.  When using
 * IMEX it performs one extra evaluation because there are more
 * implicit steps than explicit.
 *
 * The implicit portion is L-stable.
 *
 * The CFL factor/stable step size is 1.25637.
 */
class Rk3Pareschi : public ImexRungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 3rd-order 4 stage Runge-Kutta scheme devised by Pareschi and Russo."};

  Rk3Pareschi() = default;
  Rk3Pareschi(const Rk3Pareschi&) = default;
  Rk3Pareschi& operator=(const Rk3Pareschi&) = default;
  Rk3Pareschi(Rk3Pareschi&&) = default;
  Rk3Pareschi& operator=(Rk3Pareschi&&) = default;
  ~Rk3Pareschi() override = default;

  size_t order() const override;

  double stable_step() const override;

  size_t imex_order() const override;

  size_t implicit_stage_order() const override;

  WRAPPED_PUPable_decl_template(Rk3Pareschi);  // NOLINT

  explicit Rk3Pareschi(CkMigrateMessage* /*unused*/) {}

  const ButcherTableau& butcher_tableau() const override;

  const ImplicitButcherTableau& implicit_butcher_tableau() const override;
};

inline bool constexpr operator==(const Rk3Pareschi& /*lhs*/,
                                 const Rk3Pareschi& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Rk3Pareschi& lhs,
                                 const Rk3Pareschi& rhs) {
  return not(lhs == rhs);
}
}  // namespace TimeSteppers
