// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ElementActions.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/InitializeElement.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitor.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver {

/*!
 * \ingroup LinearSolverGroup
 * \brief A conjugate gradient solver for linear systems of equations \f$Ax=b\f$
 * where the operator \f$A\f$ is symmetric.
 *
 * \details The only operation we need to supply to the algorithm is the
 * result of the operation \f$A(p)\f$ (see \ref LinearSolverGroup) that in the
 * case of the conjugate gradient algorithm must be symmetric. Each invocation
 * of the `perform_step` action expects that \f$A(p)\f$ has been computed in a
 * preceding action and stored in the DataBox as
 * %db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 * db::add_tag_prefix<LinearSolver::Tags::Operand, typename
 * Metavariables::system::fields_tag>>.
 *
 * Note that the operand \f$p\f$ for which \f$A(p)\f$ needs to be computed is
 * not the field \f$x\f$ we are solving for but
 * `db::add_tag_prefix<LinearSolver::Tags::Operand, typename
 * Metavariables::system::fields_tag>`. This field is initially set to the
 * residual \f$r = b - A(x_0)\f$ where \f$x_0\f$ is the initial value of the
 * `Metavariables::system::fields_tag`.
 *
 * When the `perform_step` action is invoked after the operator action
 * \f$A(p)\f$ has been computed and stored in the DataBox, the conjugate
 * gradient algorithm implemented here will converge the field \f$x\f$ towards
 * the solution and update the operand \f$p\f$ in the process. This requires
 * two reductions over all elements that are received by a `ResidualMonitor`
 * singleton parallel component, processed, and then broadcast back to all
 * elements. The actions are implemented in the `cg_detail` namespace and
 * constitute the full algorithm in the following order:
 * 1. `PerformStep` (on elements): Compute the inner product \f$\langle p,
 * A(p)\rangle\f$ and reduce.
 * 2. `ComputeAlpha` (on `ResidualMonitor`): Compute
 * \f$\alpha=\frac{r^2}{\langle p, A(p)\rangle}\f$ and broadcast.
 * 3. `UpdateFieldValues` (on elements): Update \f$x\f$ and \f$r\f$, then
 * compute the inner product \f$\langle r, r\rangle\f$ and reduce to find the
 * new \f$r^2\f$.
 * 4. `UpdateResidual` (on `ResidualMonitor`): Store the new \f$r^2\f$ and
 * broadcast the ratio of the new and old \f$r^2\f$, as well as a termination
 * flag if the residual vanishes to a precision determined by
 * `equal_within_roundoff`.
 * 5. `UpdateOperand` (on elements): Update \f$p\f$. Stop if termination flag
 * was received.
 */
template <typename Metavariables>
struct ConjugateGradient {
  /*!
   * \brief The parallel components used by the conjugate gradient linear solver
   *
   * Uses:
   * - System:
   *   * `fields_tag`
   */
  using component_list = tmpl::list<cg_detail::ResidualMonitor<Metavariables>>;

  /*!
   * \brief Initialize the tags used by the conjugate gradient linear solver
   *
   * Uses:
   * - System:
   *   * `fields_tag`
   * - ConstGlobalCache: nothing
   *
   * With:
   * - `operand_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>`
   * - `operator_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>`
   * - `residual_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>`
   *
   * DataBox changes:
   * - Adds:
   *   * `LinearSolver::Tags::IterationId`
   *   * `Tags::Next<LinearSolver::Tags::IterationId>`
   *   * `operand_tag`
   *   * `operator_tag`
   *   * `residual_tag`
   * - Removes: nothing
   * - Modifies: nothing
   */
  using tags = cg_detail::InitializeElement<Metavariables>;

  /*!
   * \brief Perform an iteration of the conjugate gradient linear solver
   *
   * Uses:
   * - System:
   *   * `fields_tag`
   * - ConstGlobalCache: nothing
   *
   * With:
   * - `operand_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>`
   * - `residual_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>`
   *
   * DataBox changes:
   * - Adds: nothing
   * - Removes: nothing
   * - Modifies:
   *   * `LinearSolver::Tags::IterationId`
   *   * `Tags::Next<LinearSolver::Tags::IterationId>`
   *   * `fields_tag`
   *   * `residual_tag`
   *   * `operand_tag`
   */
  using perform_step = cg_detail::PerformStep;
};

}  // namespace LinearSolver
