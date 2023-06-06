// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include "Options/String.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeDelta;
namespace TimeSteppers {
template <typename T>
class ConstUntypedHistory;
template <typename T>
class MutableUntypedHistory;
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
///
/// \note The time stepper is only strong-stability-preserving for
/// time steps not exceeding 1.0, i.e., slightly less than 0.8 times
/// the stable step.
class Rk3HesthavenSsp : public TimeStepper {
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

  WRAPPED_PUPable_decl_template(Rk3HesthavenSsp);  // NOLINT

  explicit Rk3HesthavenSsp(CkMigrateMessage* /*unused*/) {}

  size_t order() const override;

  size_t error_estimate_order() const override;

  double stable_step() const override;

  uint64_t number_of_substeps() const override;

  uint64_t number_of_substeps_for_error() const override;

  size_t number_of_past_steps() const override;

  TimeStepId next_time_id(const TimeStepId& current_id,
                          const TimeDelta& time_step) const override;

  TimeStepId next_time_id_for_error(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const override;

 private:
  template <typename T>
  void update_u_impl(gsl::not_null<T*> u,
                     const MutableUntypedHistory<T>& history,
                     const TimeDelta& time_step) const;

  template <typename T>
  bool update_u_impl(gsl::not_null<T*> u, gsl::not_null<T*> u_error,
                     const MutableUntypedHistory<T>& history,
                     const TimeDelta& time_step) const;

  template <typename T>
  bool dense_update_u_impl(gsl::not_null<T*> u,
                           const ConstUntypedHistory<T>& history,
                           double time) const;

  template <typename T>
  bool can_change_step_size_impl(const TimeStepId& time_id,
                                 const ConstUntypedHistory<T>& history) const;

  TIME_STEPPER_DECLARE_OVERLOADS
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
