// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Time/TimeSteppers/ImexTimeStepper.hpp"
#include "Time/TimeSteppers/RungeKutta.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class TimeDelta;
namespace TimeSteppers {
template <typename T>
class ConstUntypedHistory;
}  // namespace TimeSteppers
/// \endcond

namespace TimeSteppers {
/*!
 * \ingroup TimeSteppersGroup
 * Intermediate base class implementing a generic IMEX Runge-Kutta
 * scheme.
 *
 * Implements most of the virtual methods of ImexTimeStepper for a
 * generic Runge-Kutta method.  Derived classes must implement the
 * requirements of the `RungeKutta` base class, as well as
 * `imex_order()`, `implicit_stage_order()`, and
 * `implicit_butcher_tableau()`.
 */
class ImexRungeKutta : public virtual RungeKutta,
                       public virtual ImexTimeStepper {
 public:
  /// Implicit part of the Butcher tableau.  Most parts of the tableau
  /// must be the same as the explicit part, and so are omitted.
  struct ImplicitButcherTableau {
    /*!
     * The coefficient matrix of the substeps.  We can only reasonably
     * support EDIRK methods (including special cases such as ESDIRK
     * and QESDIRK, see \cite Kennedy2016), so the tableau must be
     * lower-triangular with an empty first row.  As with the explicit
     * tableau, the initial blank row should be omitted.  For a
     * stiffly-accurate method, the final row must be the same as the
     * result coefficients in the explicit tableau.
     *
     * More general DIRK methods can be implemented inefficiently by
     * adding an unused initial substep to convert them to EDIRK form.
     */
    std::vector<std::vector<double>> substep_coefficients;
  };

  /*!
   * Smallest order of the intermediate result at any substep.  For
   * the methods supported by this class, this cannot exceed 2.
   */
  virtual size_t implicit_stage_order() const = 0;

  virtual const ImplicitButcherTableau& implicit_butcher_tableau() const = 0;

 private:
  template <typename T>
  void add_inhomogeneous_implicit_terms_impl(
      gsl::not_null<T*> u, const ConstUntypedHistory<T>& implicit_history,
      const TimeDelta& time_step) const;

  template <typename T>
  double implicit_weight_impl(const ConstUntypedHistory<T>& implicit_history,
                              const TimeDelta& time_step) const;

  IMEX_TIME_STEPPER_DECLARE_OVERLOADS
};
}  // namespace TimeSteppers
