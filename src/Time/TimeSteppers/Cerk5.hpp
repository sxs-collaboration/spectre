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
 * \brief A fifth order continuous-extension RK method that provides 5th-order
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
 * in \cite Owren1992 and \cite Gassner20114232. Note that \f$c_1 = 0\f$,
 * \f$s\f$ is the number of stages, and \f$\theta\f$ is the fraction of the
 * step. This is an FSAL stepper.
 *
 * The CFL factor/stable step size is 1.5961737362090775.
 */
class Cerk5 : public TimeStepper {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 5th-order continuous extension Runge-Kutta time stepper."};

  Cerk5() = default;
  Cerk5(const Cerk5&) = default;
  Cerk5& operator=(const Cerk5&) = default;
  Cerk5(Cerk5&&) = default;
  Cerk5& operator=(Cerk5&&) = default;
  ~Cerk5() override = default;

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

  WRAPPED_PUPable_decl_template(Cerk5);  // NOLINT

  explicit Cerk5(CkMigrateMessage* /*msg*/);

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

  static constexpr double a2_ = 1.0 / 6.0;
  static constexpr std::array<double, 2> a3_{{1.0 / 16.0, 3.0 / 16.0}};
  static constexpr std::array<double, 3> a4_{{0.25, -0.75, 1.0}};
  static constexpr std::array<double, 4> a5_{{-0.75, 15.0 / 4.0, -3.0, 0.5}};
  static constexpr std::array<double, 5> a6_{{369.0 / 1372.0, -243.0 / 343.0,
                                              297.0 / 343.0, 1485.0 / 9604.0,
                                              297.0 / 4802.0}};
  static constexpr std::array<double, 6> a7_{
      {-133.0 / 4512.0, 1113.0 / 6016.0, 7945.0 / 16544.0, -12845.0 / 24064.0,
       -315.0 / 24064.0, 156065.0 / 198528.0}};
  // a8 gives the final value and allows us to evaluate k_8 which we need for
  // dense output
  static constexpr std::array<double, 7> a8_{
      {83.0 / 945.0, 0.0, 248.0 / 825.0, 41.0 / 180.0, 1.0 / 36.0,
       2401.0 / 38610.0, 6016.0 / 20475.0}};

  // For the dense output coefficients the index indicates
  // `(degree of theta) - 1`.
  static constexpr std::array<double, 6> b1_{{-a8_[0], 1.0, -3292.0 / 819.0,
                                              17893.0 / 2457.0, -4969.0 / 819.0,
                                              596.0 / 315.0}};
  // b2 = {-a8_[1], 0.0, ...}
  static constexpr double b2_ = -a8_[1];
  static constexpr std::array<double, 6> b3_{{-a8_[2], 0.0, 5112.0 / 715.0,
                                              -43568.0 / 2145.0, 1344.0 / 65.0,
                                              -1984.0 / 275.0}};
  static constexpr std::array<double, 6> b4_{{-a8_[3], 0.0, -123.0 / 52.0,
                                              3161.0 / 234.0, -1465.0 / 78.0,
                                              118.0 / 15.0}};
  static constexpr std::array<double, 6> b5_{
      {-a8_[4], 0.0, -63.0 / 52.0, 1061.0 / 234.0, -413.0 / 78.0, 2.0}};
  static constexpr std::array<double, 6> b6_{
      {-a8_[5], 0.0, -40817.0 / 33462.0, 60025.0 / 50193.0, 2401.0 / 1521.0,
       -9604.0 / 6435.0}};
  static constexpr std::array<double, 6> b7_{
      {-a8_[6], 0.0, 18048.0 / 5915.0, -637696.0 / 53235.0, 96256.0 / 5915.0,
       -48128.0 / 6825.0}};
  static constexpr std::array<double, 6> b8_{
      {0.0, 0.0, -18.0 / 13.0, 75.0 / 13.0, -109.0 / 13.0, 4.0}};

  // constants for discrete error estimate
  static constexpr std::array<double, 8> e_{{-1.0 / 9.0, 0.0, 40.0 / 33.0,
                                             -7.0 / 4.0, -1.0 / 12.0,
                                             343.0 / 198.0, 0.0, 0.0}};
  static const std::array<Time::rational_t, 6> c_;
};

inline bool constexpr operator==(const Cerk5& /*lhs*/, const Cerk5& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Cerk5& /*lhs*/, const Cerk5& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
