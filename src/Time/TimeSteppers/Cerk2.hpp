// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct TimeStepId;
namespace TimeSteppers {
template <typename T>
class UntypedHistory;
}  // namespace TimeSteppers
/// \endcond

namespace TimeSteppers {
/*!
 * \ingroup TimeSteppersGroup
 * \brief A second order continuous-extension RK method that provides 2nd-order
 * dense output.
 *
 * \f{eqnarray}{
 * \frac{du}{dt} & = & \mathcal{L}(t,u).
 * \f}
 * Given a solution \f$u(t^n)=u^n\f$, this stepper computes
 * \f$u(t^{n+1})=u^{n+1}\f$ using the following equations:
 *
 * \f{align}{
 * k^{(i)} & = \mathcal{L}(t^n + c_i \Delta t,
 *                         u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} k^{(j)}),
 *                              \mbox{ } 1 \leq i \leq s,\\
 * u^{n+1}(t^n + \theta \Delta t) & = u^n + \Delta t \sum_{i=1}^{s} b_i(\theta)
 * k^{(i)}. \f}
 *
 * Here the coefficients \f$a_{ij}\f$, \f$b_i\f$, and \f$c_i\f$ are given
 * in \cite Gassner20114232. Note that \f$c_1 = 0\f$, \f$s\f$ is the number
 * of stages, and \f$\theta\f$ is the fraction of the step.
 *
 * The CFL factor/stable step size is 1.0.
 */
class Cerk2 : public TimeStepper {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 2nd order accurate continuous extension Runge-Kutta method.."};

  Cerk2() = default;
  Cerk2(const Cerk2&) = default;
  Cerk2& operator=(const Cerk2&) = default;
  Cerk2(Cerk2&&) = default;
  Cerk2& operator=(Cerk2&&) = default;
  ~Cerk2() override = default;

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

  WRAPPED_PUPable_decl_template(Cerk2);  // NOLINT

  explicit Cerk2(CkMigrateMessage* /*unused*/) {}

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

  static constexpr double a2_ = 1.0;
  static constexpr std::array<double, 2> a3_{{0.5, 0.5}};

  // For the dense output coefficients the index indicates
  // `(degree of theta)`.
  static constexpr std::array<double, 3> b1_{{-a3_[0], 1.0, -0.5}};
  static constexpr std::array<double, 3> b2_{{-a3_[1], 0.0, 0.5}};

  // constants for discrete error estimate
  static constexpr std::array<double, 2> e_{{1.0, 0.0}};
  static const std::array<Time::rational_t, 1> c_;
};

inline bool constexpr operator==(const Cerk2& /*lhs*/, const Cerk2& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Cerk2& /*lhs*/, const Cerk2& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
