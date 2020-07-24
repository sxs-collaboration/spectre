// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementActions.hpp"
#include "Utilities/TMPL.hpp"

/// Items related to the %Schwarz linear solver
///
/// \see `LinearSolver::Schwarz::Schwarz`
namespace LinearSolver::Schwarz {

/*!
 * \ingroup LinearSolverGroup
 * \brief A Schwarz-type solver for linear systems of equations \f$Ax=b\f$.
 *
 * Solving Ax=b approximately by inverting A element-wise with some overlap:
 * 1. Gather source data with overlap from neighbors
 * 2. Iteratively solve Ax=b restricted to subdomain by applying operator to
 * fields (without communication)
 *   - Possible libraries: Eigen, PETSc
 * 3. Weight result with function and sum over subdomains
 */
template <typename Metavariables, typename FieldsTag, typename OptionsGroup,
          typename SubdomainOperator, typename SubdomainPreconditioner = void,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct Schwarz {
  using operand_tag = FieldsTag;
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using options_group = OptionsGroup;

  using component_list = tmpl::list<>;
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<async_solvers::reduction_data, detail::reduction_data>>;

  using initialize_element = tmpl::list<
      async_solvers::InitializeElement<FieldsTag, OptionsGroup, SourceTag>,
      detail::InitializeElement<FieldsTag, OptionsGroup, SubdomainOperator,
                                SubdomainPreconditioner>>;

  using register_element = tmpl::list<
      async_solvers::RegisterElement<FieldsTag, OptionsGroup, SourceTag>,
      detail::RegisterElement<FieldsTag, OptionsGroup, SourceTag>>;

  template <typename ApplyOperatorActions, typename Label = OptionsGroup>
  using solve = tmpl::list<
      async_solvers::PrepareSolve<FieldsTag, OptionsGroup, SourceTag, Label>,
      detail::SendOverlapData<FieldsTag, OptionsGroup, SubdomainOperator>,
      detail::SolveSubdomain<FieldsTag, OptionsGroup, SubdomainOperator,
                             SubdomainPreconditioner>,
      detail::ReceiveOverlapSolution<FieldsTag, OptionsGroup,
                                     SubdomainOperator>,
      ApplyOperatorActions,
      async_solvers::CompleteStep<FieldsTag, OptionsGroup, SourceTag, Label>>;
};

}  // namespace LinearSolver::Schwarz
