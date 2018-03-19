// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class RungeKutta3.

#pragma once

#include <cstddef>
#include <pup.h>
#include <type_traits>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/BoundaryHistory.hpp"  // IWYU pragma: keep
#include "Time/History.hpp"          // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct TimeId;
/// \endcond

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
///
/// A "strong stability-preserving" 3rd-order Runge-Kutta time-stepper.
/// Major reference:  J. Hesthaven & T. Warburton, Nodal Discontinuous
/// Galerkin Methods. section 5.7
class RungeKutta3 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "A third-order strong stability-preserving Runge-Kutta time-stepper."};

  RungeKutta3() = default;
  RungeKutta3(const RungeKutta3&) noexcept = default;
  RungeKutta3& operator=(const RungeKutta3&) noexcept = default;
  RungeKutta3(RungeKutta3&&) noexcept = default;
  RungeKutta3& operator=(RungeKutta3&&) noexcept = default;
  ~RungeKutta3() noexcept override = default;

  template <typename Vars, typename DerivVars>
  void update_u(gsl::not_null<Vars*> u,
                gsl::not_null<History<Vars, DerivVars>*> history,
                const TimeDelta& time_step) const noexcept;

  template <typename LocalVars, typename RemoteVars, typename Coupling>
  std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
  compute_boundary_delta(
      const Coupling& coupling,
      gsl::not_null<BoundaryHistory<
          LocalVars, RemoteVars,
          std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>*>
          history,
      const TimeDelta& time_step) const noexcept;

  size_t number_of_substeps() const noexcept override;

  size_t number_of_past_steps() const noexcept override;

  bool is_self_starting() const noexcept override;

  double stable_step() const noexcept override;

  TimeId next_time_id(const TimeId& current_id,
                      const TimeDelta& time_step) const noexcept override;

  WRAPPED_PUPable_decl_template(RungeKutta3);  // NOLINT

  explicit RungeKutta3(CkMigrateMessage* /*unused*/) noexcept {}

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& p) noexcept override {  // NOLINT
    TimeStepper::Inherit::pup(p);
  }
};

inline bool constexpr operator==(const RungeKutta3& /*lhs*/,
                                 const RungeKutta3& /*rhs*/) noexcept {
  return true;
}

inline bool constexpr operator!=(const RungeKutta3& /*lhs*/,
                                 const RungeKutta3& /*rhs*/) noexcept {
  return false;
}

template <typename Vars, typename DerivVars>
void RungeKutta3::update_u(
    const gsl::not_null<Vars*> u,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  const size_t substep = history->size() - 1;
  const auto& vars = (history->end() - 1).value();
  const auto& dt_vars = (history->end() - 1).derivative();
  const auto& U0 = history->begin().value();

  switch (substep) {
    case 0: {
      // from (5.32) of Hesthaven
      // v^(1) = u^n + dt*RHS(u^n,t^n)
      // On entry V = u^n, U0 = u^n, rhs0 = RHS(u^n,t^n),
      // time = t^n
      *u += time_step.value() * dt_vars;
      // On exit v = v^(1), time = t^n + dt
      break;
    }
    case 1: {
      // from (5.32) of Hesthaven
      // v^(2) = (1/4)*( 3*u^n + v^(1) + dt*RHS(v^(1),t^n + dt) )
      // On entry V = v^(1), U0 = u^n, rhs0 = RHS(v^(1),t^n + dt),
      // time = t^n + dt
      *u += 0.25 * (3.0 * (U0 - vars) + time_step.value() * dt_vars);
      // On exit v = v^(2), time = t^n + (1/2)*dt
      break;
    }
    case 2: {
      // from (5.32) of Hesthaven
      // u^(n+1) = (1/3)*( u^n + 2*v^(2) + 2*dt*RHS(v^(2),t^n + (1/2)*dt) )
      // On entry V = v^(2), U0 = u^n, rhs0 = RHS(v^(2),t^n + (1/2)*dt),
      // time = t^n + (1/2)*dt
      *u += (1.0 / 3.0) * (U0 - vars + 2.0 * time_step.value() * dt_vars);
      // On exit v = u^(n+1), time = t^n + dt
      break;
    }
    default:
      ERROR("Bad substep value in RK3: " << substep);
  }

  // Clean up old history
  if (history->size() == number_of_substeps()) {
    history->mark_unneeded(history->end());
  }
}

template <typename LocalVars, typename RemoteVars, typename Coupling>
std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
RungeKutta3::compute_boundary_delta(
    const Coupling& coupling,
    const gsl::not_null<BoundaryHistory<
        LocalVars, RemoteVars,
        std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>*>
        history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(history->local_size() == history->remote_size(),
         "Inconsistent history sizes for global time step method");
  const size_t substep = history->local_size() - 1;
  auto dt_vars = history->coupling(
      coupling, history->local_end() - 1, history->remote_end() - 1);

  // These are the coefficients of the dt_vars terms in the volume
  // algorithm.
  double prefactor;
  switch (substep) {
    case 0:
      prefactor = 1.0;
      break;
    case 1:
      prefactor = 0.25;
      break;
    case 2:
      prefactor = 2.0 / 3.0;
      break;
    default:
      ERROR("Bad substep value in RK3: " << substep);
  }

  // Clean up old history
  if (history->local_size() == number_of_substeps()) {
    history->local_mark_unneeded(history->local_end());
    history->remote_mark_unneeded(history->remote_end());
  }

  return prefactor * time_step.value() * dt_vars;
}
}  // namespace TimeSteppers
