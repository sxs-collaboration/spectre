// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class RungeKutta3.

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct TimeStepId;
namespace TimeSteppers {
template <typename T>
class UntypedHistory;
}  // namespace TimeSteppers
/// \endcond

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
///
/// A "strong stability-preserving" 3rd-order Runge-Kutta
/// time-stepper, as described in \cite HesthavenWarburton section
/// 5.7.
///
/// The CFL factor/stable step size is 1.25637266330916.
class RungeKutta3 : public TimeStepper {
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

  uint64_t number_of_substeps() const override;

  uint64_t number_of_substeps_for_error() const override;

  size_t number_of_past_steps() const override;

  double stable_step() const override;

  TimeStepId next_time_id(const TimeStepId& current_id,
                          const TimeDelta& time_step) const override;

  TimeStepId next_time_id_for_error(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const override;

  WRAPPED_PUPable_decl_template(RungeKutta3);  // NOLINT

  explicit RungeKutta3(CkMigrateMessage* /*unused*/) {}

 private:
  template <typename T>
  void update_u_impl(gsl::not_null<T*> u,
                     gsl::not_null<UntypedHistory<T>*> history,
                     const TimeDelta& time_step) const;

  template <typename T>
  bool update_u_impl(gsl::not_null<T*> u, gsl::not_null<T*> u_error,
                     gsl::not_null<UntypedHistory<T>*> history,
                     const TimeDelta& time_step) const;

  template <typename T>
  bool dense_update_u_impl(gsl::not_null<T*> u,
                           const UntypedHistory<T>& history, double time) const;

  template <typename T>
  bool can_change_step_size_impl(const TimeStepId& time_id,
                                 const UntypedHistory<T>& history) const;

  TIME_STEPPER_DECLARE_OVERLOADS
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
