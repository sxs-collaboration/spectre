// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class RungeKutta3.

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
class RungeKutta3 : public RungeKutta {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A third-order strong stability-preserving Runge-Kutta time-stepper."};

  RungeKutta3() = default;
  RungeKutta3(const RungeKutta3&) = default;
  RungeKutta3& operator=(const RungeKutta3&) = default;
  RungeKutta3(RungeKutta3&&) = default;
  RungeKutta3& operator=(RungeKutta3&&) = default;
  ~RungeKutta3() override = default;

  size_t order() const override;

  size_t error_estimate_order() const override;

  double stable_step() const override;

  WRAPPED_PUPable_decl_template(RungeKutta3);  // NOLINT

  explicit RungeKutta3(CkMigrateMessage* /*unused*/) {}

 private:
  const ButcherTableau& butcher_tableau() const override;
};

inline bool constexpr operator==(const RungeKutta3& /*lhs*/,
                                 const RungeKutta3& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const RungeKutta3& /*lhs*/,
                                 const RungeKutta3& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
