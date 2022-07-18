// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Rational.hpp"

/// \cond
class TimeDelta;
namespace TimeSteppers {
template <typename T>
class UntypedHistory;
}  // namespace TimeSteppers
/// \endcond

namespace TimeSteppers {
/*!
 * \ingroup TimeSteppersGroup
 * Intermediate base class implementing a generic Runge-Kutta scheme.
 *
 * Implements most of the virtual methods of TimeStepper for a generic
 * Runge-Kutta method.  From the TimeStepper interface, derived
 * classes need only implement `order`, `error_estimate_order`, and
 * `stable_step`, and should not include the `TIME_STEPPER_*` macros.
 * All other methods are implemented in terms of a Butcher tableau
 * returned by the `butcher_tableau` function and, optionally, another
 * from the `error_tableau` if the scheme's error estimation requires
 * additional substeps.
 */
class RungeKutta : public TimeStepper {
 public:
  struct ButcherTableau {
    /*!
     * The times of the substeps, excluding the initial time step.
     * Often called \f$c\f$ in the literature.
     */
    std::vector<Rational> substep_times;
    /*!
     * The coefficient matrix of the substeps.  Do not include the
     * initial empty row or the coefficients for the full step.  Often
     * called \f$A\f$ in the literature.
     */
    std::vector<std::vector<double>> substep_coefficients;
    /*!
     * The coefficients for the final result.  Often called \f$b\f$ in
     * the literature.
     */
    std::vector<double> result_coefficients;
    /*!
     * The coefficients for an error estimate.  Often called
     * \f$b^*\f$ or \f$\hat{b}\f$ in the literature.
     *
     * Some RK methods need extra steps to compute error measures (in
     * addition to or in place of using an FSAL optimization).  For
     * such steppers, this vector can be left empty for the main
     * Butcher tableau and a separate tableau for error estimation can
     * be returned from the `error_tableau` function.
     */
    std::vector<double> error_coefficients;
    /*!
     * Coefficient polynomials for dense output.  Each entry is the
     * coefficients of a polynomial that will be evaluated with the
     * fraction of the way through the step at which output is
     * desired:
     *
     * \f{equation}{
     *   y = y_0 + \sum_i y'_i \sum_j p_{ij} x^j \qquad 0 \le x \le 1
     * \f}
     *
     * The derivative at the start of the next step is available as an
     * additional substep after the final real substep.  Trailing zero
     * polynomials can be omitted.
     */
    std::vector<std::vector<double>> dense_coefficients;
  };

  uint64_t number_of_substeps() const override;

  uint64_t number_of_substeps_for_error() const override;

  size_t number_of_past_steps() const override;

  TimeStepId next_time_id(const TimeStepId& current_id,
                          const TimeDelta& time_step) const override;

  TimeStepId next_time_id_for_error(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const override;

 protected:
  virtual const ButcherTableau& butcher_tableau() const = 0;

  virtual const ButcherTableau& error_tableau() const;

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
}  // namespace TimeSteppers
