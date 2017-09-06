// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TimeStepper.

#pragma once

#include <deque>
#include <tuple>

#include "Time/Time.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"

/// \ingroup TimeSteppersGroup
///
/// Holds classes that take time steps.
namespace TimeSteppers {
class AdamsBashforthN;
class RungeKutta3;
}  // namespace TimeSteppers

namespace TimeStepper_detail {
DEFINE_FAKE_VIRTUAL(update_u)
}  // namespace TimeStepper_detail

/// \ingroup TimeSteppersGroup
///
/// Abstract base class for TimeSteppers.
class TimeStepper /*: public Factory<TimeStepper>*/ {
 public:
  using Inherit = TimeStepper_detail::FakeVirtualInherit_update_u<TimeStepper>;
  static std::string class_id() noexcept { return "TimeStepper"; }
  using creatable_classes = typelist<
      TimeSteppers::AdamsBashforthN,
      TimeSteppers::RungeKutta3>;

  /// \cond HIDDEN_SYMBOLS
  TimeStepper() noexcept = default;
  TimeStepper(const TimeStepper&) noexcept = default;
  TimeStepper& operator=(const TimeStepper&) noexcept = default;
  TimeStepper(TimeStepper&&) noexcept = default;
  TimeStepper& operator=(TimeStepper&&) noexcept = default;
  virtual ~TimeStepper() noexcept = default;
  /// \endcond

  /// Add the change for the current substep to u and return the
  /// substep size, i.e., the time delta to the next derivative
  /// evaluation.  New values should be pushed onto the end of the
  /// history when evaluated and obsolete values should be removed
  /// from the front, in a manner similar to a queue.
  template <typename Vars, typename DerivVars>
  TimeDelta update_u(
      const gsl::not_null<Vars*> u,
      const std::deque<std::tuple<Vars, DerivVars, TimeDelta>>& history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper_detail::fake_virtual_update_u<creatable_classes>(
        this, u, history, time_step);
  }

  /// Number of substeps in this TimeStepper
  virtual size_t number_of_substeps() const noexcept = 0;

  /// Number of past time steps needed for multi-step method
  virtual size_t number_of_past_steps() const noexcept = 0;

  /// Whether or not the method is self-starting
  virtual bool is_self_starting() const noexcept = 0;

  /// Rough estimate of the maximum step size this method can take
  /// stably as a multiple of the step for Euler's method.
  virtual double stable_step() const noexcept = 0;
};

#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
