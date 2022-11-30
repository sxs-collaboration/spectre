// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/TimeSteppers/RungeKutta.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
///
/// A "strong stability-preserving" 3rd-order Runge-Kutta
/// time-stepper, as described in \cite HesthavenWarburton section
/// 5.7.
///
/// The CFL factor/stable step size is 1.25637266330916.
class Rk3HesthavenSsp : public RungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A third-order strong stability-preserving Runge-Kutta time-stepper."};

  Rk3HesthavenSsp() = default;
  Rk3HesthavenSsp(const Rk3HesthavenSsp&) = default;
  Rk3HesthavenSsp& operator=(const Rk3HesthavenSsp&) = default;
  Rk3HesthavenSsp(Rk3HesthavenSsp&&) = default;
  Rk3HesthavenSsp& operator=(Rk3HesthavenSsp&&) = default;
  ~Rk3HesthavenSsp() override = default;

  size_t order() const override;

  size_t error_estimate_order() const override;

  double stable_step() const override;

  WRAPPED_PUPable_decl_template(Rk3HesthavenSsp);  // NOLINT

  explicit Rk3HesthavenSsp(CkMigrateMessage* /*unused*/) {}

 private:
  const ButcherTableau& butcher_tableau() const override;
};

inline bool constexpr operator==(const Rk3HesthavenSsp& /*lhs*/,
                                 const Rk3HesthavenSsp& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Rk3HesthavenSsp& /*lhs*/,
                                 const Rk3HesthavenSsp& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
