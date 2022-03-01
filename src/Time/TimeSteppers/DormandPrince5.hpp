// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DormandPrince5.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <pup.h>
#include <tuple>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
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

/*!
 * \ingroup TimeSteppersGroup
 *
 * The standard 5th-order Dormand-Prince time stepping method, given e.g. in
 * Sec. 7.2 of \cite NumericalRecipes.
 *
 * \f{eqnarray}{
 * \frac{du}{dt} & = & \mathcal{L}(t,u).
 * \f}
 * Given a solution \f$u(t^n)=u^n\f$, this stepper computes
 * \f$u(t^{n+1})=u^{n+1}\f$ using the following equations:
 *
 * \f{align}{
 * k^{(1)} & = dt \mathcal{L}(t^n, u^n),\\
 * k^{(i)} & = dt \mathcal{L}(t^n + c_i dt,
 *                              u^n + \sum_{j=1}^{i-1} a_{ij} k^{(j)}),
 *                              \mbox{ } 2 \leq i \leq 6,\\
 * u^{n+1} & = u^n + \sum_{i=1}^{6} b_i k^{(i)}.
 * \f}
 *
 * Here the coefficients \f$a_{ij}\f$, \f$b_i\f$, and \f$c_i\f$ are given
 * in e.g. Sec. 7.2 of \cite NumericalRecipes. Note that \f$c_1 = 0\f$.
 *
 * The CFL factor/stable step size is 1.6532839463174733.
 */
class DormandPrince5 : public TimeStepper {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "The standard Dormand-Prince 5th-order time stepper."};

  DormandPrince5() = default;
  DormandPrince5(const DormandPrince5&) = default;
  DormandPrince5& operator=(const DormandPrince5&) = default;
  DormandPrince5(DormandPrince5&&) = default;
  DormandPrince5& operator=(DormandPrince5&&) = default;
  ~DormandPrince5() override = default;

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

  WRAPPED_PUPable_decl_template(DormandPrince5);  // NOLINT

  explicit DormandPrince5(CkMigrateMessage* /*unused*/) {}

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

  // Coefficients from the Dormand-Prince 5 Butcher tableau (e.g. Sec. 7.2
  // of \cite NumericalRecipes).
  static constexpr std::array<double, 1> a2_{{0.2}};
  static constexpr std::array<double, 2> a3_{{3.0 / 40.0, 9.0 / 40.0}};
  static constexpr std::array<double, 3> a4_{
      {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0}};
  static constexpr std::array<double, 4> a5_{
      {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0}};
  static constexpr std::array<double, 5> a6_{{9017.0 / 3168.0, -355.0 / 33.0,
                                              46732.0 / 5247.0, 49.0 / 176.0,
                                              -5103.0 / 18656.0}};
  static constexpr std::array<double, 6> b_{{35.0 / 384.0, 0.0, 500.0 / 1113.0,
                                             125.0 / 192.0, -2187.0 / 6784.0,
                                             11.0 / 84.0}};
  // Coefficients for the embedded method, for generating an error measure
  // during adaptive stepping (e.g. Sec. 7.2 of \cite NumericalRecipes).
  static constexpr std::array<double, 7> b_alt_{
      {5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
       -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0}};

  static const std::array<Time::rational_t, 6> c_;

  // Coefficients for dense output, taken from Sec. 7.2 of
  // \cite NumericalRecipes
  static constexpr std::array<double, 7> d_{
      {-12715105075.0 / 11282082432.0, 0.0, 87487479700.0 / 32700410799.0,
       -10690763975.0 / 1880347072.0, 701980252875.0 / 199316789632.0,
       -1453857185.0 / 822651844.0, 69997945.0 / 29380423.0}};
};

inline bool constexpr operator==(const DormandPrince5& /*lhs*/,
                                 const DormandPrince5& /*rhs*/) {
  return true;
}

inline bool constexpr operator!=(const DormandPrince5& /*lhs*/,
                                 const DormandPrince5& /*rhs*/) {
  return false;
}
}  // namespace TimeSteppers
