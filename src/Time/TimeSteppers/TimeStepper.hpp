// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TimeStepper.

#pragma once

#include <deque>
#include <tuple>
#include <type_traits>
#include <vector>

#include "Parallel/CharmPupable.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/Time.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"

struct TimeId;

/// \ingroup TimeSteppersGroup
///
/// Holds classes that take time steps.
namespace TimeSteppers {
class AdamsBashforthN;
class RungeKutta3;
}  // namespace TimeSteppers

namespace TimeStepper_detail {
DEFINE_FAKE_VIRTUAL(compute_boundary_delta)
DEFINE_FAKE_VIRTUAL(update_u)
}  // namespace TimeStepper_detail

/// \ingroup TimeSteppersGroup
///
/// Abstract base class for TimeSteppers.
class TimeStepper : public PUP::able {
 public:
  using Inherit =
      TimeStepper_detail::FakeVirtualInherit_compute_boundary_delta<
          TimeStepper_detail::FakeVirtualInherit_update_u<TimeStepper>>;
  using creatable_classes =
      tmpl::list<TimeSteppers::AdamsBashforthN, TimeSteppers::RungeKutta3>;

  WRAPPED_PUPable_abstract(TimeStepper);  // NOLINT

  /// \cond HIDDEN_SYMBOLS
  TimeStepper() = default;
  TimeStepper(const TimeStepper&) noexcept = default;
  TimeStepper& operator=(const TimeStepper&) noexcept = default;
  TimeStepper(TimeStepper&&) noexcept = default;
  TimeStepper& operator=(TimeStepper&&) noexcept = default;
  ~TimeStepper() noexcept override = default;
  /// \endcond

  /// Add the change for the current substep to u.
  template <typename Vars, typename DerivVars>
  void update_u(
      const gsl::not_null<Vars*> u,
      const gsl::not_null<TimeSteppers::History<Vars, DerivVars>*> history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper_detail::fake_virtual_update_u<creatable_classes>(
        this, u, history, time_step);
  }

  /// \brief Compute the change in a boundary quantity due to the
  /// coupling on the interface.
  ///
  /// The coupling function `coupling` should take the local and
  /// remote flux data and compute the derivative of the boundary
  /// quantity.  These values may be used to form a linear combination
  /// internally, so the result should have appropriate mathematical
  /// operators defined to allow that.
  template <typename LocalVars, typename RemoteVars, typename Coupling>
  std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
  compute_boundary_delta(
      const Coupling& coupling,
      const gsl::not_null<TimeSteppers::BoundaryHistory<
          LocalVars, RemoteVars,
          std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>*>
          history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper_detail::fake_virtual_compute_boundary_delta<
        creatable_classes>(this, coupling, history, time_step);
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

  /// The TimeId after the current substep
  virtual TimeId next_time_id(const TimeId& current_id,
                              const TimeDelta& time_step) const noexcept = 0;
};

#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
