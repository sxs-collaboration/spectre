// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TimeStepper.

#pragma once

#include <deque>
#include <tuple>
#include <type_traits>

#include "Options/Factory.hpp"
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
DEFINE_FAKE_VIRTUAL(compute_boundary_delta)
DEFINE_FAKE_VIRTUAL(needed_boundary_history)
DEFINE_FAKE_VIRTUAL(needed_history)
DEFINE_FAKE_VIRTUAL(update_u)
}  // namespace TimeStepper_detail

/// \ingroup TimeSteppersGroup
///
/// Abstract base class for TimeSteppers.
class TimeStepper : public Factory<TimeStepper> {
 public:
  using Inherit =
      TimeStepper_detail::FakeVirtualInherit_compute_boundary_delta<
          TimeStepper_detail::FakeVirtualInherit_needed_boundary_history<
              TimeStepper_detail::FakeVirtualInherit_needed_history<
                  TimeStepper_detail::FakeVirtualInherit_update_u<
                      TimeStepper>>>>;
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
      const std::deque<std::tuple<Time, Vars, DerivVars>>& history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper_detail::fake_virtual_update_u<creatable_classes>(
        this, u, history, time_step);
  }

  /// \brief Compute the change in a boundary quantity due to the
  /// coupling on the interface.
  ///
  /// The `history` is a vector representing the different sides of
  /// the interface, with the local side first.  Each entry in that
  /// vector is the time history of a side, similar to the argument to
  /// `update_u`.
  ///
  /// The coupling function `coupling` should compute the derivative
  /// of the boundary quantity from the side data, supplied as a
  /// `std::vector<std::reference_wrapper<const FluxVars>>`.  These
  /// values may be used to form a linear combination internally, so
  /// `BoundaryVars` should have appropriate mathematical operators
  /// defined to allow that.
  template <typename BoundaryVars, typename FluxVars, typename Coupling>
  BoundaryVars compute_boundary_delta(
      const Coupling& coupling,
      const std::vector<std::deque<std::tuple<Time, BoundaryVars, FluxVars>>>&
          history,
      const TimeDelta& time_step) const noexcept {
    static_assert(
        std::is_convertible<
            std::decay_t<std::result_of_t<const Coupling&(
                const std::vector<std::reference_wrapper<const FluxVars>>&)>>,
            BoundaryVars>::value,
        "Coupling function returns wrong type");
    return TimeStepper_detail::fake_virtual_compute_boundary_delta<
        creatable_classes>(this, coupling, history, time_step);
  }

  /// Return iterator to the first entry in `history` that is still
  /// needed after the current step.  Entries up to that point must be
  /// removed before the next call to `update_u`.
  template <typename Vars, typename DerivVars>
  typename std::deque<std::tuple<Time, Vars, DerivVars>>::const_iterator
  needed_history(const std::deque<std::tuple<Time, Vars, DerivVars>>& history)
      const noexcept {
    return TimeStepper_detail::fake_virtual_needed_history<creatable_classes>(
        this, history);
  }

  /// Return iterators to the first entry for each element of
  /// `history` that is still needed after the current step.  Entries
  /// up to that point must be removed before the next call to
  /// `compute_boundary_delta`.
  template <typename BoundaryVars, typename FluxVars>
  typename std::vector<typename std::deque<
      std::tuple<Time, BoundaryVars, FluxVars>>::const_iterator>
  needed_boundary_history(
      const std::vector<
          std::deque<std::tuple<Time, BoundaryVars, FluxVars>>>& history,
      const TimeDelta& time_step) const noexcept {
    return TimeStepper_detail::fake_virtual_needed_boundary_history<
        creatable_classes>(this, history, time_step);
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
