// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <pup.h>
#include <type_traits>

#include "Parallel/CharmPupable.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeDelta;
namespace TimeSteppers {
class AdamsBashforthN;  // IWYU pragma: keep
template <typename LocalVars, typename RemoteVars, typename CouplingResult>
class BoundaryHistory;
template <typename Vars, typename DerivVars>
class History;
}  // namespace TimeSteppers
/// \endcond

/// \ingroup TimeSteppersGroup
///
/// Holds classes that take time steps.
namespace TimeSteppers {
class AdamsBashforthN;  // IWYU pragma: keep
class DormandPrince5;
class RungeKutta3;  // IWYU pragma: keep
class RungeKutta4;
}  // namespace TimeSteppers

namespace TimeStepper_detail {
DEFINE_FAKE_VIRTUAL(can_change_step_size)
DEFINE_FAKE_VIRTUAL(dense_update_u)
DEFINE_FAKE_VIRTUAL(update_u)
}  // namespace TimeStepper_detail

/// \ingroup TimeSteppersGroup
///
/// Abstract base class for TimeSteppers.
class TimeStepper : public PUP::able {
 public:
  using Inherit = TimeStepper_detail::FakeVirtualInherit_can_change_step_size<
      TimeStepper_detail::FakeVirtualInherit_dense_update_u<
          TimeStepper_detail::FakeVirtualInherit_update_u<TimeStepper>>>;
  using creatable_classes =
      tmpl::list<TimeSteppers::AdamsBashforthN, TimeSteppers::DormandPrince5,
                 TimeSteppers::RungeKutta3, TimeSteppers::RungeKutta4>;

  WRAPPED_PUPable_abstract(TimeStepper);  // NOLINT

  /// Add the change for the current substep to u.
  template <typename Vars, typename DerivVars>
  void update_u(
      const gsl::not_null<Vars*> u,
      const gsl::not_null<TimeSteppers::History<Vars, DerivVars>*> history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper_detail::fake_virtual_update_u<creatable_classes>(
        this, u, history, time_step);
  }

  /// Add the change for the current substep to u; report the error measure when
  /// available.
  ///
  /// For a substep method, the error measure will only be available on full
  /// steps. For a multistep method, the error measure will only be available
  /// when a sufficient number of steps are available in the `history` to
  /// compare two orders of step. Whenever the error measure is unavailable,
  /// `u_error` is unchanged and the function return is `false`.
  template <typename Vars, typename ErrVars, typename DerivVars>
  bool update_u(
      const gsl::not_null<Vars*> u, const gsl::not_null<ErrVars*> u_error,
      const gsl::not_null<TimeSteppers::History<Vars, DerivVars>*> history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper_detail::fake_virtual_update_u<creatable_classes>(
        this, u, u_error, history, time_step);
  }

  /// Compute the solution value at a time between steps.  To evaluate
  /// at a time within a given step, this must be called before the
  /// step is completed but after any intermediate substeps have been
  /// taken.
  template <typename Vars, typename DerivVars>
  void dense_update_u(const gsl::not_null<Vars*> u,
                      const TimeSteppers::History<Vars, DerivVars>& history,
                      const double time) const noexcept {
    return TimeStepper_detail::fake_virtual_dense_update_u<creatable_classes>(
        this, u, history, time);
  }

  /// The convergence order of the stepper
  virtual size_t order() const noexcept = 0;

  /// The convergence order of the stepper error measure
  virtual size_t error_estimate_order() const noexcept = 0;

  /// Number of substeps in this TimeStepper
  virtual uint64_t number_of_substeps() const noexcept = 0;

  /// Number of substeps in this TimeStepper when providing an error measure for
  /// adaptive time-stepping
  ///
  /// \details Certain substep methods (e.g. embedded RK4(3)) require additional
  /// steps when providing an error measure of the integration.
  virtual uint64_t number_of_substeps_for_error() const noexcept = 0;

  /// Number of past time steps needed for multi-step method
  virtual size_t number_of_past_steps() const noexcept = 0;

  /// Rough estimate of the maximum step size this method can take
  /// stably as a multiple of the step for Euler's method.
  virtual double stable_step() const noexcept = 0;

  /// The TimeStepId after the current substep
  virtual TimeStepId next_time_id(
      const TimeStepId& current_id,
      const TimeDelta& time_step) const noexcept = 0;

  /// The TimeStepId after the current substep when providing an error measure
  /// for adaptive time-stepping.
  ///
  /// Certain substep methods (e.g. embedded RK4(3)) require additional
  /// steps when providing an error measure of the integration.
  virtual TimeStepId next_time_id_for_error(
      const TimeStepId& current_id,
      const TimeDelta& time_step) const noexcept = 0;

  /// Whether a change in the step size is allowed before taking
  /// a step.
  template <typename Vars, typename DerivVars>
  bool can_change_step_size(
      const TimeStepId& time_id,
      const TimeSteppers::History<Vars, DerivVars>& history) const noexcept {
    return TimeStepper_detail::fake_virtual_can_change_step_size<
        creatable_classes>(this, time_id, history);
  }
};

// LtsTimeStepper cannot be split out into its own file because the
// LtsTimeStepper -> TimeStepper -> AdamsBashforthN -> LtsTimeStepper
// include loop causes AdamsBashforthN to be defined before its base
// class if LtsTimeStepper is included first.
namespace LtsTimeStepper_detail {
DEFINE_FAKE_VIRTUAL(boundary_dense_output)
DEFINE_FAKE_VIRTUAL(compute_boundary_delta)
}  // namespace LtsTimeStepper_detail

/// \ingroup TimeSteppersGroup
///
/// Base class for TimeSteppers with local time-stepping support,
/// derived from TimeStepper.
class LtsTimeStepper : public TimeStepper::Inherit {
 public:
  using Inherit =
      LtsTimeStepper_detail::FakeVirtualInherit_boundary_dense_output<
          LtsTimeStepper_detail::FakeVirtualInherit_compute_boundary_delta<
              LtsTimeStepper>>;
  // When you add a class here, remember to add it to TimeStepper as well.
  using creatable_classes = tmpl::list<TimeSteppers::AdamsBashforthN>;

  WRAPPED_PUPable_abstract(LtsTimeStepper);  // NOLINT

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
    return LtsTimeStepper_detail::fake_virtual_compute_boundary_delta<
        creatable_classes>(this, coupling, history, time_step);
  }

  template <typename LocalVars, typename RemoteVars, typename Coupling>
  std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
  boundary_dense_output(
      const Coupling& coupling,
      const TimeSteppers::BoundaryHistory<
          LocalVars, RemoteVars,
          std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>& history,
      const double time) const noexcept {
    return LtsTimeStepper_detail::fake_virtual_boundary_dense_output<
        creatable_classes>(this, coupling, history, time);
  }

  /// Substep LTS integrators are not supported, so this is always 1.
  uint64_t number_of_substeps() const noexcept final { return 1; }

  /// Substep LTS integrators are not supported, so this is always 1.
  uint64_t number_of_substeps_for_error() const noexcept final { return 1; }

  TimeStepId next_time_id_for_error(
      const TimeStepId& current_id,
      const TimeDelta& time_step) const noexcept final {
    return next_time_id(current_id, time_step);
  }

  /// \cond
  // FakeVirtual forces derived classes to override the fake virtual
  // methods.  Here the base class method is actually what we want
  // because we are not a most-derived class, so we forward to the
  // TimeStepper version.
  template <typename Vars, typename DerivVars>
  void update_u(
      const gsl::not_null<Vars*> u,
      const gsl::not_null<TimeSteppers::History<Vars, DerivVars>*> history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper::update_u(u, history, time_step);
  }

  template <typename Vars, typename ErrVars, typename DerivVars>
  bool update_u(
      const gsl::not_null<Vars*> u, const gsl::not_null<ErrVars*> u_error,
      const gsl::not_null<TimeSteppers::History<Vars, DerivVars>*> history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper::update_u(u, u_error, history, time_step);
  }

  template <typename Vars, typename DerivVars>
  bool can_change_step_size(
      const TimeStepId& time_id,
      const TimeSteppers::History<Vars, DerivVars>& history) const noexcept {
    return TimeStepper::can_change_step_size(time_id, history);
  }
  /// \endcond
};


#include "Time/TimeSteppers/AdamsBashforthN.hpp"  // IWYU pragma: keep
#include "Time/TimeSteppers/DormandPrince5.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"  // IWYU pragma: keep
#include "Time/TimeSteppers/RungeKutta4.hpp"  // IWYU pragma: keep
