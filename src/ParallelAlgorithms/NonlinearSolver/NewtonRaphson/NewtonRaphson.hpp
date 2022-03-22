// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/ElementActions.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/ResidualMonitor.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace NonlinearSolver {
/// Items related to the NewtonRaphson nonlinear solver
namespace newton_raphson {

/*!
 * \brief A Newton-Raphson correction scheme for nonlinear systems of equations
 * \f$A_\mathrm{nonlinear}(x)=b\f$.
 *
 * We can use a correction scheme to solve a nonlinear problem
 * \f$A_\mathrm{nonlinear}(x)=b\f$ by repeatedly solving a linearization of it.
 * A Newton-Raphson scheme iteratively refines an initial guess \f$x_0\f$ by
 * repeatedly solving the linearized problem
 *
 * \f{equation}
 * \frac{\delta A_\mathrm{nonlinear}}{\delta x}(x_k) \delta x_k =
 * b-A_\mathrm{nonlinear}(x_k) \equiv r_k
 * \f}
 *
 * for the correction \f$\delta x_k\f$ and then updating the solution as
 * \f$x_{k+1}=x_k + \delta x_k\f$.
 *
 * The operations we need to supply to the algorithm are the nonlinear operator
 * \f$A_\mathrm{nonlinear}(x)\f$ and a linear solver for the linearized problem
 * \f$\frac{\delta A_\mathrm{nonlinear}}{\delta x}(x_k) \delta x_k = r_k\f$.
 * Each step of the algorithm expects that \f$A_\mathrm{nonlinear}(x)\f$ is
 * computed and stored in the DataBox as
 * `db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, FieldsTag>`.
 * To perform a solve, add the `solve` action list to an array parallel
 * component. Pass the actions that compute \f$A_\mathrm{nonlinear}(x)\f$ as
 * the first template parameter to `solve`. As the second template parameter,
 * pass the action list that performs a linear solve of the linearized operator
 * \f$\frac{\delta A_\mathrm{nonlinear}}{\delta x}(x)\f$ for the field
 * \f$\delta x\f$ (the `linear_solver_fields_tag`) sourced by
 * \f$r\f$ (the `linear_solver_source_tag`). You will find suitable iterative
 * linear solvers in the `LinearSolver` namespace. The third template parameter
 * allows you to pass any further actions you wish to run in each step of the
 * algorithm (such as observations). If you add the `solve` action list multiple
 * times, use the fourth template parameter to label each solve with a different
 * type.
 *
 * \par Globalization:
 * This nonlinear solver supports a line-search (or "backtracking")
 * globalization. If a step does not sufficiently decrease the residual (see
 * `NonlinearSolver::OptionTags::SufficientDecrease` for details on the
 * sufficient decrease condition), the step length is reduced until the residual
 * sufficiently decreases or the maximum number of globalization steps is
 * reached. The reduction in step length is determined by the minimum of a
 * quadratic (first globalization step) or cubic (subsequent globalization
 * steps) polynomial interpolation according to Algorithm A6.1.3 in
 * \cite DennisSchnabel (p. 325) (see
 * `NonlinearSolver::newton_raphson::next_step_length`). Alternatives to a
 * line-search globalization, such as a trust-region globalization or more
 * sophisticated nonlinear preconditioning techniques (see e.g. \cite Brune2015
 * for an overview), are not currently implemented.
 *
 * \par Array sections
 * This nonlinear solver supports running over a subset of the elements in the
 * array parallel component (see `Parallel::Section`). Set the
 * `ArraySectionIdTag` template parameter to restrict the solver to elements in
 * that section. Only a single section must be associated with the
 * `ArraySectionIdTag`. The default is `void`, which means running over all
 * elements in the array. Note that the actions in the `SolveLinearization`
 * list passed to `solve` will _not_ be restricted to run only on section
 * elements, so all elements in the array may participate in preconditioning
 * (see LinearSolver::multigrid::Multigrid).
 */
template <typename Metavariables, typename FieldsTag, typename OptionsGroup,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>,
          typename ArraySectionIdTag = void>
struct NewtonRaphson {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;
  using source_tag = SourceTag;

  using operand_tag = fields_tag;
  using linear_solver_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using linear_solver_source_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;

  using component_list = tmpl::list<
      detail::ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>;

  using initialize_element =
      detail::InitializeElement<FieldsTag, OptionsGroup, SourceTag>;

  using register_element = tmpl::list<>;

  template <typename ApplyNonlinearOperator, typename SolveLinearization,
            typename ObserveActions = tmpl::list<>,
            typename Label = OptionsGroup>
  using solve = tmpl::list<
      detail::PrepareSolve<FieldsTag, OptionsGroup, Label, ArraySectionIdTag>,
      ApplyNonlinearOperator, ObserveActions,
      detail::SendInitialResidualMagnitude<FieldsTag, OptionsGroup, Label,
                                           ArraySectionIdTag>,
      detail::ReceiveInitialHasConverged<FieldsTag, OptionsGroup, Label,
                                         ArraySectionIdTag>,
      detail::PrepareStep<FieldsTag, OptionsGroup, Label, ArraySectionIdTag>,
      SolveLinearization,
      detail::PerformStep<FieldsTag, OptionsGroup, Label, ArraySectionIdTag>,
      ApplyNonlinearOperator,
      detail::ContributeToResidualMagnitudeReduction<FieldsTag, OptionsGroup,
                                                     Label, ArraySectionIdTag>,
      detail::Globalize<FieldsTag, OptionsGroup, Label, ArraySectionIdTag>,
      ObserveActions,
      detail::CompleteStep<FieldsTag, OptionsGroup, Label, ArraySectionIdTag>>;
};

}  // namespace newton_raphson
}  // namespace NonlinearSolver
