// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <cstddef>
#include <iosfwd>

#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace LinearSolver {

/*!
 * \ingroup LinearSolverGroup
 * \brief Criteria that determine the linear solve has converged
 *
 * \details Most criteria are based on the residual magnitude
 * \f$|r_k|=|b-Ax_k|\f$, where \f$x_k\f$ denotes the numerical solution after
 * completion of iteration \f$k\f$ (see the \ref LinearSolverGroup
 * documentation, `LinearSolver::Tags::Residual` and
 * `LinearSolver::Tags::Magnitude`).
 *
 * The following criteria are implemented, ordered from highest to lowest
 * priority:
 *
 * - AbsoluteResidual: Matches if the residual has reached this magnitude.
 * - RelativeResidual: Matches if the residual has decreased by this factor,
 * relative to the start of the first linear solver iteration.
 * - MaxIterations: Matches if the number of iterations exceeds this limit.
 *
 * \note The smallest possible residual magnitude the linear solver can reach is
 * the product between the machine epsilon and the condition number of the
 * linear operator that is being inverted. Smaller residuals are numerical
 * artifacts. Requiring an absolute or relative residual below this limit will
 * likely lead to termination by `MaxIterations`.
 *
 * \note Remember that when the linear operator \f$A\f$ corresponds to a PDE
 * discretization, decreasing the linear solver residual below the
 * discretization error will not improve the numerical solution any further.
 * I.e. the error \f$e_k=x_k-x_\mathrm{analytic}\f$ to an analytic solution
 * will be dominated by the linear solver residual at first, but even if the
 * discretization \f$Ax_k=b\f$ was exactly solved after some iteration \f$k\f$,
 * the discretization residual
 * \f$Ae_k=b-Ax_\mathrm{analytic}=r_\mathrm{discretization}\f$ would still
 * remain. Therefore, ideally choose the absolute or relative residual criteria
 * based on an estimate of the discretization residual.
 */
struct ConvergenceCriteria {
  static constexpr OptionString help =
      "The linear solver terminates when any of these criteria is matched.";

  struct MaxIterations {
    using type = size_t;
    static constexpr OptionString help = {
        "The number of iterations exceeds this limit."};
    static type lower_bound() noexcept { return 0; }
  };

  struct AbsoluteResidual {
    using type = double;
    static constexpr OptionString help = {
        "The residual has reached this magnitude."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct RelativeResidual {
    using type = double;
    static constexpr OptionString help = {
        "The residual has decreased by this factor."};
    static type lower_bound() noexcept { return 0.; }
    static type upper_bound() noexcept { return 1.; }
  };

  using options = tmpl::list<MaxIterations, AbsoluteResidual, RelativeResidual>;

  ConvergenceCriteria() = default;
  ConvergenceCriteria(size_t max_iterations_in, double absolute_residual_in,
                      double relative_residual_in) noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

  size_t max_iterations{};
  double absolute_residual{};
  double relative_residual{};
};

bool operator==(const ConvergenceCriteria& lhs,
                const ConvergenceCriteria& rhs) noexcept;
bool operator!=(const ConvergenceCriteria& lhs,
                const ConvergenceCriteria& rhs) noexcept;

/*!
 * \brief The reason the linear solver has converged.
 *
 * \see LinearSolver::ConvergenceCriteria
 */
enum class ConvergenceReason {
  MaxIterations,
  AbsoluteResidual,
  RelativeResidual
};

std::ostream& operator<<(std::ostream& os,
                         const ConvergenceReason& convergence_reason) noexcept;

/*!
 * \brief Determine whether the \p convergence_criteria are met.
 *
 * \note This function assumes the \p iteration_id is that of the next, but
 * not yet performed step. For instance, a `MaxIteration` criterion of 1 will
 * match if the \p iteration_id is 1 or higher, since the first iteration
 * (with id 0) has been completed. At this point, also the \p residual_magnitude
 * reflects the state of the algorithm after completion of the first iteration.
 * The `initial_residual_magnitude` always refers to the state before the first
 * iteration has begun.
 *
 * \returns a `LinearSolver::ConvergenceReason` if the criteria are met, or
 * `boost::none` otherwise.
 */
boost::optional<ConvergenceReason> convergence_criteria_match(
    const ConvergenceCriteria& convergence_criteria, size_t iteration_id,
    double residual_magnitude, double initial_residual_magnitude) noexcept;

/*!
 * \brief Signals convergence of the linear solver.
 *
 * \details Evaluates to `true` if the linear solver has converged and no
 * further iterations should be performed. In this case, the `reason()` member
 * function provides more information. If `false`, calling `reason()` is an
 * error.
 *
 * The stream operator provides a human-readable description of the convergence
 * status.
 *
 * This type default-constructs to a state that signals the linear solver has
 * not yet converged.
 */
struct HasConverged {
 public:
  HasConverged() = default;
  /*!
   * \brief Determine whether the \p convergence_criteria are met by means of
   * `LinearSolver::convergence_criteria_match`.
   */
  HasConverged(const ConvergenceCriteria& convergence_criteria,
               size_t iteration_id, double residual_magnitude,
               double initial_residual_magnitude) noexcept;

  explicit operator bool() const noexcept { return static_cast<bool>(reason_); }

  /*!
   * \brief The reason the linear solver has converged.
   *
   * \warning Calling this function is an error if the linear solver has not yet
   * converged.
   */
  ConvergenceReason reason() const noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

  friend bool operator==(const HasConverged& lhs,
                         const HasConverged& rhs) noexcept;
  friend bool operator!=(const HasConverged& lhs,
                         const HasConverged& rhs) noexcept;

  friend std::ostream& operator<<(std::ostream& os,
                                  const HasConverged& has_converged) noexcept;

 private:
  boost::optional<ConvergenceReason> reason_{boost::none};
  ConvergenceCriteria convergence_criteria_{};
  size_t iteration_id_{};
  double residual_magnitude_{};
  double initial_residual_magnitude_{};
};

}  // namespace LinearSolver
