// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class RungeKutta3.

#pragma once

#include <algorithm>
#include <deque>
#include <functional>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/MakeWithValue.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
///
/// A "strong stability-preserving" 3rd-order Runge-Kutta time-stepper.
/// Major reference:  J. Hesthaven & T. Warburton, Nodal Discontinuous
/// Galerkin Methods. section 5.7
class RungeKutta3 : public TimeStepper::Inherit {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString_t help = {
      "A third-order strong stability-preserving Runge-Kutta time-stepper."};
  explicit RungeKutta3(const OptionContext& /*context*/) noexcept {}

  RungeKutta3() noexcept = default;
  RungeKutta3(const RungeKutta3&) noexcept = default;
  RungeKutta3& operator=(const RungeKutta3&) noexcept = default;
  RungeKutta3(RungeKutta3&&) noexcept = default;
  RungeKutta3& operator=(RungeKutta3&&) noexcept = default;
  ~RungeKutta3() noexcept override = default;

  template <typename Vars, typename DerivVars>
  TimeDelta update_u(
      gsl::not_null<Vars*> u,
      const std::deque<std::tuple<Time, Vars, DerivVars>>& history,
      const TimeDelta& time_step) const noexcept;

  template <typename BoundaryVars, typename FluxVars, typename Coupling>
  BoundaryVars compute_boundary_delta(
      const Coupling& coupling,
      const std::vector<std::deque<std::tuple<Time, BoundaryVars, FluxVars>>>&
          history,
      const TimeDelta& time_step) const noexcept;

  template <typename Vars, typename DerivVars>
  typename std::deque<std::tuple<Time, Vars, DerivVars>>::const_iterator
  needed_history(const std::deque<std::tuple<Time, Vars, DerivVars>>& history)
      const noexcept;

  size_t number_of_substeps() const noexcept override;

  size_t number_of_past_steps() const noexcept override;

  bool is_self_starting() const noexcept override;

  double stable_step() const noexcept override;

 private:
  template <typename Vars, typename UnusedData, typename DerivVars>
  TimeDelta step_work(
    gsl::not_null<Vars*> u,
    const std::deque<std::tuple<Time, Vars, UnusedData>>& history,
    const TimeDelta& time_step,
    const DerivVars& dt_vars) const noexcept;
};

template <typename Vars, typename DerivVars>
TimeDelta RungeKutta3::update_u(
    const gsl::not_null<Vars*> u,
    const std::deque<std::tuple<Time, Vars, DerivVars>>& history,
    const TimeDelta& time_step) const noexcept {
  return step_work(u, history, time_step, std::get<2>(history.back()));
}

template <typename BoundaryVars, typename FluxVars, typename Coupling>
BoundaryVars RungeKutta3::compute_boundary_delta(
    const Coupling& coupling,
    const std::vector<std::deque<std::tuple<Time, BoundaryVars, FluxVars>>>&
        history,
    const TimeDelta& time_step) const noexcept {
  std::vector<std::reference_wrapper<const FluxVars>> coupling_args;
  std::transform(history.begin(), history.end(),
                 std::inserter(coupling_args, coupling_args.begin()),
                 [](const auto& side) {
                   return std::cref(std::get<2>(side.back()));
                 });

  auto u = make_with_value<BoundaryVars>(std::get<1>(history[0][0]), 0.);
  step_work(make_not_null(&u), history[0], time_step, coupling(coupling_args));
  return u;
}

template <typename Vars, typename UnusedData, typename DerivVars>
TimeDelta RungeKutta3::step_work(
    const gsl::not_null<Vars*> u,
    const std::deque<std::tuple<Time, Vars, UnusedData>>& history,
    const TimeDelta& time_step,
    const DerivVars& dt_vars) const noexcept {
  const size_t substep = history.size() - 1;
  const auto& vars = std::get<1>(history.back());
  const auto& U0 = std::get<1>(history[0]);

  switch (substep) {
    case 0: {
      // from (5.32) of Hesthaven
      // v^(1) = u^n + dt*RHS(u^n,t^n)
      // On entry V = u^n, U0 = u^n, rhs0 = RHS(u^n,t^n),
      // time = t^n
      *u += time_step.value() * dt_vars;
      return time_step;
      // On exit v = v^(1), time = t^n + dt
    }
    case 1: {
      // from (5.32) of Hesthaven
      // v^(2) = (1/4)*( 3*u^n + v^(1) + dt*RHS(v^(1),t^n + dt) )
      // On entry V = v^(1), U0 = u^n, rhs0 = RHS(v^(1),t^n + dt),
      // time = t^n + dt
      *u += 0.25 * (3.0 * (U0 - vars) + time_step.value() * dt_vars);
      return -time_step / 2;
      // On exit v = v^(2), time = t^n + (1/2)*dt
    }
    case 2: {
      // from (5.32) of Hesthaven
      // u^(n+1) = (1/3)*( u^n + 2*v^(2) + 2*dt*RHS(v^(2),t^n + (1/2)*dt) )
      // On entry V = v^(2), U0 = u^n, rhs0 = RHS(v^(2),t^n + (1/2)*dt),
      // time = t^n + (1/2)*dt
      *u += (1.0 / 3.0) * (U0 - vars + 2.0 * time_step.value() * dt_vars);
      return time_step / 2;
      // On exit v = u^(n+1), time = t^n + dt
    }
    default:
      ERROR("Bad substep value in RK3: " << substep);
  }
}

template <typename Vars, typename DerivVars>
typename std::deque<std::tuple<Time, Vars, DerivVars>>::const_iterator
RungeKutta3::needed_history(
    const std::deque<std::tuple<Time, Vars, DerivVars>>& history) const
    noexcept {
  const bool at_step_end = history.size() == number_of_substeps();
  return at_step_end ? history.end() : history.begin();
}

}  // namespace TimeSteppers
