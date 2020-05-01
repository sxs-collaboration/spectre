// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/InitializeElement.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ResidualMonitor.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver {

/*!
 * \ingroup LinearSolverGroup
 * \brief A GMRES solver for nonsymmetric linear systems of equations
 * \f$Ax=b\f$.
 *
 * \details The only operation we need to supply to the algorithm is the
 * result of the operation \f$A(p)\f$ (see \ref LinearSolverGroup). Each
 * invocation of the `perform_step` action expects that \f$A(q)\f$ has been
 * computed in a preceding action and stored in the DataBox as
 * %db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 * db::add_tag_prefix<LinearSolver::Tags::Operand, FieldsTag>>.
 *
 * Note that the operand \f$q\f$ for which \f$A(q)\f$ needs to be computed is
 * not the field \f$x\f$ we are solving for but
 * `db::add_tag_prefix<LinearSolver::Tags::Operand, FieldsTag>`. This field is
 * initially set to the residual \f$q_0 = b - A(x_0)\f$ where \f$x_0\f$ is the
 * initial value of the `FieldsTag`.
 *
 * When the `perform_step` action is invoked after the operator action
 * \f$A(q)\f$ has been computed and stored in the DataBox, the GMRES algorithm
 * implemented here will converge the field \f$x\f$ towards the solution and
 * update the operand \f$q\f$ in the process. This requires reductions over
 * all elements that are received by a `ResidualMonitor` singleton parallel
 * component, processed, and then broadcast back to all elements. Since the
 * reductions are performed to find a vector that is orthogonal to those used in
 * previous steps, the number of reductions increases linearly with iterations.
 * No restarting mechanism is currently implemented. The actions are implemented
 * in the `gmres_detail` namespace and constitute the full algorithm in the
 * following order:
 * 1. `PerformStep` (on elements): Start an Arnoldi orthogonalization by
 * computing the inner product between \f$A(q)\f$ and the first of the
 * previously determined set of orthogonal vectors.
 * 2. `StoreOrthogonalization` (on `ResidualMonitor`): Keep track of the
 * computed inner product in a Hessenberg matrix, then broadcast.
 * 3. `OrthogonalizeOperand` (on elements): Proceed with the Arnoldi
 * orthogonalization by computing inner products and reducing to
 * `StoreOrthogonalization` on the `ResidualMonitor` until the new orthogonal
 * vector is constructed. Then compute its magnitude and reduce.
 * 4. `StoreFinalOrthogonalization` (on `ResidualMonitor`): Perform a QR
 * decomposition of the Hessenberg matrix to produce a residual vector.
 * Broadcast to `NormalizeOperandAndUpdateField` along with a termination
 * flag if the `LinearSolver::Tags::ConvergenceCriteria` are met.
 * 5. `NormalizeOperandAndUpdateField` (on elements): Set the operand \f$q\f$ as
 * the new orthogonal vector and normalize. Use the residual vector and the set
 * of orthogonal vectors to determine the solution \f$x\f$.
 *
 * \see ConjugateGradient for a linear solver that is more efficient when the
 * linear operator \f$A\f$ is symmetric.
 */
template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct Gmres {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;

  /// Apply the linear operator to this tag in each iteration
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;

  /*!
   * \brief The parallel components used by the GMRES linear solver
   */
  using component_list = tmpl::list<
      gmres_detail::ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>;

  /*!
   * \brief Initialize the tags used by the GMRES linear solver.
   *
   * Since we have not started iterating yet, we initialize the state _before_
   * the first iteration. So `LinearSolver::Tags::IterationId` is undefined at
   * this point and `Tags::Next<LinearSolver::Tags::IterationId>` is the initial
   * step number. Invoke `prepare_step` to advance the state to the first
   * iteration.
   *
   * \warning This action involves a blocking reduction, so it is a global
   * synchronization point.
   *
   * With:
   * - `initial_fields_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Initial, FieldsTag>`
   * - `operand_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Operand, FieldsTag>`
   * - `operator_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>`
   * - `orthogonalization_iteration_id_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
   * LinearSolver::Tags::IterationId>`
   * - `basis_history_tag` =
   * `LinearSolver::Tags::KrylovSubspaceBasis<FieldsTag>`
   *
   * DataBox changes:
   * - Adds:
   *   * `LinearSolver::Tags::IterationId`
   *   * `Tags::Next<LinearSolver::Tags::IterationId>`
   *   * `initial_fields_tag`
   *   * `orthogonalization_iteration_id_tag`
   *   * `basis_history_tag`
   *   * `LinearSolver::Tags::HasConverged`
   * - Removes: nothing
   * - Modifies:
   *   * `operand_tag`
   *
   * \note The `operand_tag` must already be present in the DataBox and is set
   * to its initial value here. It is typically added to the DataBox by the
   * system, which uses it to compute the `operator_tag` in each step. Also the
   * `operator_tag` is typically added to the DataBox by the system, but does
   * not need to be initialized until it is computed for the first time in the
   * first step of the algorithm.
   */
  using initialize_element =
      gmres_detail::InitializeElement<FieldsTag, OptionsGroup>;

  /*!
   * \brief Reset the linear solver to its initial state.
   *
   * Uses:
   * - System:
   *   * `fields_tag`
   *
   * With:
   * - `initial_fields_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Initial, fields_tag>`
   * - `operand_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>`
   * - `orthogonalization_iteration_id_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
   * LinearSolver::Tags::IterationId>`
   * - `basis_history_tag` =
   * `LinearSolver::Tags::KrylovSubspaceBasis<fields_tag>`
   *
   * DataBox changes:
   * - Adds: nothing
   * - Removes: nothing
   * - Modifies:
   *   * `LinearSolver::Tags::IterationId`
   *   * `initial_fields_tag`
   *   * `orthogonalization_iteration_id_tag`
   *   * `basis_history_tag`
   *   * `LinearSolver::Tags::HasConverged`
   *   * `operand_tag`
   *
   * \see `initialize_element`
   */
  using prepare_solve = gmres_detail::PrepareSolve<FieldsTag, OptionsGroup>;

  // Compile-time interface for observers
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<observe_detail::reduction_data>>;

  /*!
   * \brief Advance the linear solver to the next iteration.
   *
   * DataBox changes:
   * - Adds: nothing
   * - Removes: nothing
   * - Modifies:
   *   * `LinearSolver::Tags::IterationId`
   *   * `Tags::Next<LinearSolver::Tags::IterationId>`
   *   * `orthogonalization_iteration_id_tag`
   */
  using prepare_step = gmres_detail::PrepareStep<FieldsTag, OptionsGroup>;

  /*!
   * \brief Perform an iteration of the GMRES linear solver.
   *
   * \warning This action involves a blocking reduction, so it is a global
   * synchronization point.
   *
   * With:
   * - `operand_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Operand, FieldsTag>`
   * - `orthogonalization_iteration_id_tag` =
   * `db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
   * LinearSolver::Tags::IterationId>`
   * - `basis_history_tag` =
   * `LinearSolver::Tags::KrylovSubspaceBasis<FieldsTag>`
   *
   * DataBox changes:
   * - Adds: nothing
   * - Removes: nothing
   * - Modifies:
   *   * `FieldsTag`
   *   * `operand_tag`
   *   * `orthogonalization_iteration_id_tag`
   *   * `basis_history_tag`
   *   * `LinearSolver::Tags::HasConverged`
   */
  using perform_step = gmres_detail::PerformStep<FieldsTag, OptionsGroup>;
};

}  // namespace LinearSolver
