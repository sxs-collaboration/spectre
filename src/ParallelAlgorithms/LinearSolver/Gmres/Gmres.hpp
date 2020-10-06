// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/InitializeElement.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ResidualMonitor.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// Items related to the GMRES linear solver
///
/// \see `LinearSolver::gmres::Gmres`
namespace LinearSolver::gmres {

/*!
 * \ingroup LinearSolverGroup
 * \brief A GMRES solver for nonsymmetric linear systems of equations
 * \f$Ax=b\f$.
 *
 * \details The only operation we need to supply to the algorithm is the
 * result of the operation \f$A(p)\f$ (see \ref LinearSolverGroup). Each step of
 * the algorithm expects that \f$A(q)\f$ is computed and stored in the DataBox
 * as `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>`.
 * To perform a solve, add the `solve` action list to an array parallel
 * component. Pass the actions that compute \f$A(q)\f$, as well as any further
 * actions you wish to run in each step of the algorithm, as the first template
 * parameter to `solve`. If you add the `solve` action list multiple times, use
 * the second template parameter to label each solve with a different type.
 *
 * This linear solver supports preconditioning. Enable preconditioning by
 * setting the `Preconditioned` template parameter to `true`. If you do, run a
 * preconditioner (e.g. another parallel linear solver) in each step. The
 * preconditioner should approximately solve the linear problem \f$A(q)=b\f$
 * where \f$q\f$ is the `operand_tag` and \f$b\f$ is the
 * `preconditioner_source_tag`. Make sure the tag
 * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>`
 * is updated with the preconditioned result in each step of the algorithm, i.e.
 * that it is \f$A(q)\f$ where \f$q\f$ is the preconditioner's approximate
 * solution to \f$A(q)=b\f$.
 *
 * Note that the operand \f$q\f$ for which \f$A(q)\f$ needs to be computed is
 * not the field \f$x\f$ we are solving for but
 * `db::add_tag_prefix<LinearSolver::Tags::Operand, FieldsTag>`. This field is
 * initially set to the residual \f$q_0 = b - A(x_0)\f$ where \f$x_0\f$ is the
 * initial value of the `FieldsTag`.
 *
 * When the algorithm step is performed after the operator action \f$A(q)\f$ has
 * been computed and stored in the DataBox, the GMRES algorithm implemented here
 * will converge the field \f$x\f$ towards the solution and update the operand
 * \f$q\f$ in the process. This requires reductions over all elements that are
 * received by a `ResidualMonitor` singleton parallel component, processed, and
 * then broadcast back to all elements. Since the reductions are performed to
 * find a vector that is orthogonal to those used in previous steps, the number
 * of reductions increases linearly with iterations. No restarting mechanism is
 * currently implemented. The actions are implemented in the `gmres::detail`
 * namespace and constitute the full algorithm in the following order:
 * 1. `PerformStep` (on elements): Start an Arnoldi orthogonalization by
 * computing the inner product between \f$A(q)\f$ and the first of the
 * previously determined set of orthogonal vectors.
 * 2. `StoreOrthogonalization` (on `ResidualMonitor`): Keep track of the
 * computed inner product in a Hessenberg matrix, then broadcast.
 * 3. `OrthogonalizeOperand` (on elements): Proceed with the Arnoldi
 * orthogonalization by computing inner products and reducing to
 * `StoreOrthogonalization` on the `ResidualMonitor` until the new orthogonal
 * vector is constructed. Then compute its magnitude and reduce.
 * 4. `StoreOrthogonalization` (on `ResidualMonitor`): Perform a QR
 * decomposition of the Hessenberg matrix to produce a residual vector.
 * Broadcast to `NormalizeOperandAndUpdateField` along with a termination
 * flag if the `Convergence::Tags::Criteria` are met.
 * 5. `NormalizeOperandAndUpdateField` (on elements): Set the operand \f$q\f$ as
 * the new orthogonal vector and normalize. Use the residual vector and the set
 * of orthogonal vectors to determine the solution \f$x\f$.
 *
 * \see ConjugateGradient for a linear solver that is more efficient when the
 * linear operator \f$A\f$ is symmetric.
 */
template <typename Metavariables, typename FieldsTag, typename OptionsGroup,
          bool Preconditioned,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct Gmres {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;
  using source_tag = SourceTag;
  static constexpr bool preconditioned = Preconditioned;

  /// Apply the linear operator to this tag in each iteration
  using operand_tag = std::conditional_t<
      Preconditioned,
      db::add_tag_prefix<
          LinearSolver::Tags::Preconditioned,
          db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>>,
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>>;

  /// Invoke a linear solver on the `operand_tag` sourced by the
  /// `preconditioner_source_tag` before applying the operator in each step
  using preconditioner_source_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;

  /*!
   * \brief The parallel components used by the GMRES linear solver
   */
  using component_list = tmpl::list<
      detail::ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>;

  using initialize_element =
      detail::InitializeElement<FieldsTag, OptionsGroup, Preconditioned>;

  using register_element = tmpl::list<>;

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<observe_detail::reduction_data>>;

  template <typename ApplyOperatorActions, typename Label = OptionsGroup>
  using solve = tmpl::list<
      detail::PrepareSolve<FieldsTag, OptionsGroup, Preconditioned, Label,
                           SourceTag>,
      detail::NormalizeInitialOperand<FieldsTag, OptionsGroup, Preconditioned,
                                      Label>,
      detail::PrepareStep<FieldsTag, OptionsGroup, Preconditioned, Label>,
      ApplyOperatorActions,
      detail::PerformStep<FieldsTag, OptionsGroup, Preconditioned, Label>,
      detail::OrthogonalizeOperand<FieldsTag, OptionsGroup, Preconditioned,
                                   Label>,
      detail::NormalizeOperandAndUpdateField<FieldsTag, OptionsGroup,
                                             Preconditioned, Label>>;
};

}  // namespace LinearSolver::gmres
