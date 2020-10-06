// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/InitializeElement.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitor.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "Utilities/TMPL.hpp"

/// Items related to the conjugate gradient linear solver
///
/// \see `LinearSolver::cg::ConjugateGradient`
namespace LinearSolver::cg {

/*!
 * \ingroup LinearSolverGroup
 * \brief A conjugate gradient solver for linear systems of equations \f$Ax=b\f$
 * where the operator \f$A\f$ is symmetric.
 *
 * \details The only operation we need to supply to the algorithm is the
 * result of the operation \f$A(p)\f$ (see \ref LinearSolverGroup) that in the
 * case of the conjugate gradient algorithm must be symmetric. Each step of the
 * algorithm expects that \f$A(p)\f$ is computed and stored in the DataBox as
 * %db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 * db::add_tag_prefix<LinearSolver::Tags::Operand, FieldsTag>>. To perform a
 * solve, add the `solve` action list to an array parallel component. Pass the
 * actions that compute \f$A(q)\f$, as well as any further actions you wish to
 * run in each step of the algorithm, as the first template parameter to
 * `solve`. If you add the `solve` action list multiple times, use the second
 * template parameter to label each solve with a different type.
 *
 * Note that the operand \f$p\f$ for which \f$A(p)\f$ needs to be computed is
 * not the field \f$x\f$ we are solving for but
 * `db::add_tag_prefix<LinearSolver::Tags::Operand, FieldsTag>`. This field is
 * initially set to the residual \f$r = b - A(x_0)\f$ where \f$x_0\f$ is the
 * initial value of the `FieldsTag`.
 *
 * When the algorithm step is performed after the operator action \f$A(p)\f$ has
 * been computed and stored in the DataBox, the conjugate gradient algorithm
 * implemented here will converge the field \f$x\f$ towards the solution and
 * update the operand \f$p\f$ in the process. This requires two reductions over
 * all elements that are received by a `ResidualMonitor` singleton parallel
 * component, processed, and then broadcast back to all elements. The actions
 * are implemented in the `cg::detail` namespace and constitute the full
 * algorithm in the following order:
 * 1. `PerformStep` (on elements): Compute the inner product \f$\langle p,
 * A(p)\rangle\f$ and reduce.
 * 2. `ComputeAlpha` (on `ResidualMonitor`): Compute
 * \f$\alpha=\frac{r^2}{\langle p, A(p)\rangle}\f$ and broadcast.
 * 3. `UpdateFieldValues` (on elements): Update \f$x\f$ and \f$r\f$, then
 * compute the inner product \f$\langle r, r\rangle\f$ and reduce to find the
 * new \f$r^2\f$.
 * 4. `UpdateResidual` (on `ResidualMonitor`): Store the new \f$r^2\f$ and
 * broadcast the ratio of the new and old \f$r^2\f$, as well as a termination
 * flag if the `Convergence::Tags::Criteria` are met.
 * 5. `UpdateOperand` (on elements): Update \f$p\f$.
 *
 * \see Gmres for a linear solver that can invert nonsymmetric operators
 * \f$A\f$.
 */
template <typename Metavariables, typename FieldsTag, typename OptionsGroup,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct ConjugateGradient {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;
  using source_tag = SourceTag;

  /// Apply the linear operator to this tag in each iteration
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;

  /*!
   * \brief The parallel components used by the conjugate gradient linear solver
   */
  using component_list = tmpl::list<
      detail::ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>;

  using initialize_element = detail::InitializeElement<FieldsTag, OptionsGroup>;

  using register_element = tmpl::list<>;

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<observe_detail::reduction_data>>;

  template <typename ApplyOperatorActions, typename Label = OptionsGroup>
  using solve = tmpl::list<
      detail::PrepareSolve<FieldsTag, OptionsGroup, Label, SourceTag>,
      detail::InitializeHasConverged<FieldsTag, OptionsGroup, Label>,
      ApplyOperatorActions, detail::PerformStep<FieldsTag, OptionsGroup, Label>,
      detail::UpdateFieldValues<FieldsTag, OptionsGroup, Label>,
      detail::UpdateOperand<FieldsTag, OptionsGroup, Label>>;
};

}  // namespace LinearSolver::cg
