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
 * \brief A fourth order continuous-extension RK method that provides 4th-order
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
 * The CFL factor/stable step size is 1.4367588951002057.
 */
class Cerk4 : public TimeStepper {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A 4th-order continuous extension Runge-Kutta time stepper."};

  Cerk4() = default;
  Cerk4(const Cerk4&) = default;
  Cerk4& operator=(const Cerk4&) = default;
  Cerk4(Cerk4&&) = default;
  Cerk4& operator=(Cerk4&&) = default;
  ~Cerk4() override = default;

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

  WRAPPED_PUPable_decl_template(Cerk4);  // NOLINT

  explicit Cerk4(CkMigrateMessage* /*msg*/);

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
  static constexpr std::array<double, 2> a3_{{44.0 / 1369.0, 363.0 / 1369.0}};
  static constexpr std::array<double, 3> a4_{
      {3388.0 / 4913.0, -8349.0 / 4913.0, 8140.0 / 4913.0}};
  static constexpr std::array<double, 4> a5_{
      {-36764.0 / 408375.0, 767.0 / 1125.0, -32708.0 / 136125.0,
       210392.0 / 408375.0}};
  static constexpr std::array<double, 5> a6_{
      {1697.0 / 18876.0, 0.0, 50653.0 / 116160.0, 299693.0 / 1626240.0,
       3375.0 / 11648.0}};

  // For the dense output coefficients the index indicates
  // `(degree of theta) - 1`.
  static constexpr std::array<double, 5> b1_{{-a6_[0], 1.0, -104217.0 / 37466.0,
                                              1806901.0 / 618189.0,
                                              -866577.0 / 824252.0}};
  // b2 = {-a8_[1], 0.0, ...}
  static constexpr double b2_ = -a6_[1];
  static constexpr std::array<double, 5> b3_{{-a6_[2], 0.0, 861101.0 / 230560.0,
                                              -2178079.0 / 380424.0,
                                              12308679.0 / 5072320.0}};
  static constexpr std::array<double, 5> b4_{{-a6_[3], 0.0, -63869.0 / 293440.0,
                                              6244423.0 / 5325936.0,
                                              -7816583.0 / 10144640.0}};
  static constexpr std::array<double, 5> b5_{
      {-a6_[4], 0.0, -1522125.0 / 762944.0, 982125.0 / 190736.0,
       -624375.0 / 217984.0}};
  static constexpr std::array<double, 5> b6_{
      {0.0, 0.0, 165.0 / 131.0, -461.0 / 131.0, 296.0 / 131.0}};

  // constants for discrete error estimate
  static constexpr std::array<double, 6> e_{
      {101.0 / 363.0, 0.0, -1369.0 / 14520.0, 11849.0 / 14520.0, 0.0, 0.0}};
  static const std::array<Time::rational_t, 4> c_;
};

inline bool constexpr operator==(const Cerk4& /*lhs*/, const Cerk4& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const Cerk4& /*lhs*/, const Cerk4& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
