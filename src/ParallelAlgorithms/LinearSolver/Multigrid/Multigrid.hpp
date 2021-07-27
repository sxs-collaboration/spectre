// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ObserveVolumeData.hpp"
#include "Utilities/TMPL.hpp"

/// Items related to the multigrid linear solver
///
/// \see `LinearSolver::multigrid::Multigrid`
namespace LinearSolver::multigrid {

/// A label indicating the pre-smoothing step in a V-cycle multigrid algorithm,
/// i.e. the smoothing step before sending the residual to the coarser (parent)
/// grid
struct VcycleDownLabel {};

/// A label indicating the post-smoothing step in a V-cycle multigrid algorithm,
/// i.e. the smoothing step before sending the correction to the finer (child)
/// grid
struct VcycleUpLabel {};

/*!
 * \brief A V-cycle geometric multgrid solver for linear equations \f$Ax = b\f$
 *
 * This linear solver iteratively corrects an initial guess \f$x_0\f$ by
 * restricting the residual \f$b - Ax\f$ to a series of coarser grids, solving
 * for a correction on the coarser grids, and then prolongating (interpolating)
 * the correction back to the finer grids. The solves on grids with different
 * scales can very effectively solve large-scale modes in the solution, which
 * Krylov-type linear solvers such as GMRES or Conjugate Gradients typically
 * struggle with. Therefore, a multigrid solver can be an effective
 * preconditioner for Krylov-type linear solvers (see
 * `LinearSolver::gmres::Gmres` and `LinearSolver::cg::ConjugateGradient`). See
 * \cite Briggs2000jp for an introduction to multigrid methods.
 *
 * \par Grid hierarchy
 * This geometric multigrid solver relies on a strategy to coarsen the
 * computational grid in a way that removes small-scale modes. We currently
 * h-coarsen the domain, meaning that we create multigrid levels by successively
 * combining two elements into one along every dimension of the grid. We only
 * p-coarsen the grid in the sense that we choose the smaller of the two
 * polynomial degrees when combining elements. This strategy follows
 * \cite Vincent2019qpd. See `LinearSolver::multigrid::ElementsAllocator` and
 * `LinearSolver::multigrid::coarsen` for the code that creates the multigrid
 * hierarchy.
 *
 * \par Inter-mesh operators
 * The algorithm relies on operations that project data between grids. Residuals
 * are projected from finer to coarser grids ("restriction") and solutions are
 * projected from coarser to finer grids ("prolongation"). We use the standard
 * \f$L_2\f$-projections implemented in
 * `Spectral::projection_matrix_child_to_parent` and
 * `Spectral::projection_matrix_parent_to_child`, where "child" means the finer
 * grid and "parent" means the coarser grid. Note that the residuals \f$b -
 * Ax\f$ may or may not already include a mass matrix and Jacobian factors,
 * depending on the implementation of the linear operator \f$A\f$. To account
 * for this, provide a tag as the `ResidualIsMassiveTag` that holds a `bool`.
 * See section 3 in \cite Fortunato2019jl for details on the inter-mesh
 * operators.
 *
 * \par Smoother
 * On every level of the grid hierarchy the multigrid solver relies on a
 * "smoother" that performs an approximate linear solve on that level. The
 * smoother can be any linear solver that solves the `smooth_fields_tag` for the
 * source in the `smooth_source_tag`. Useful smoothers are asynchronous linear
 * solvers that parallelize well, such as `LinearSolver::Schwarz::Schwarz`. The
 * multigrid algorithm doesn't assume anything about the smoother, but requires
 * only that the `PreSmootherActions` and the `PostSmootherActions` passed to
 * `solve` leave the `smooth_fields_tag` in a state that represents an
 * approximate solution to the `smooth_source_tag`, and that
 * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 * smooth_fields_tag>` is left up-to-date as well. Here's an example of setting
 * up a smoother for the multigrid solver:
 *
 * \snippet Test_MultigridAlgorithm.cpp setup_smoother
 *
 * The smoother can be used to construct an action list like this:
 *
 * \snippet Test_MultigridAlgorithm.cpp action_list
 *
 * \par Algorithm overview
 * Every iteration of the multigrid algorithm performs a V-cycle over the grid
 * hierarchy. One V-cycle consists of first "going down" the grid hierarchy,
 * from the finest to successively coarser grids, smoothing on every level
 * ("pre-smoothing"), and then "going up" the grid hierarchy again, smoothing on
 * every level again ("post-smoothing"). When going down, the algorithm projects
 * the remaining residual of the smoother to the next-coarser grid, setting it
 * as the source for the smoother on the coarser grid. When going up again, the
 * algorithm projects the solution of the smoother to the next-finer grid,
 * adding it to the solution on the finer grid as a correction. The bottom-most
 * coarsest grid (the "tip" of the V-cycle) skips the post-smoothing, so the
 * result of the pre-smoother is immediately projected up to the finer grid. On
 * the top-most finest grid (the "original" grid that represents the overall
 * solution) the algorithm applies the smoothing and the corrections from the
 * coarser grids directly to the solution fields.
 */
template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename ResidualIsMassiveTag,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct Multigrid {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;
  using source_tag = SourceTag;

  using operand_tag = FieldsTag;

  using smooth_source_tag = source_tag;
  using smooth_fields_tag = fields_tag;

  using component_list = tmpl::list<>;

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<async_solvers::reduction_data>>;

  using initialize_element = tmpl::list<
      async_solvers::InitializeElement<FieldsTag, OptionsGroup, SourceTag>,
      detail::InitializeElement<Dim, FieldsTag, OptionsGroup, SourceTag>>;

  using register_element =
      tmpl::list<async_solvers::RegisterElement<FieldsTag, OptionsGroup,
                                                SourceTag, Tags::IsFinestGrid>,
                 observers::Actions::RegisterWithObservers<
                     detail::RegisterWithVolumeObserver<OptionsGroup>>>;

  template <typename PreSmootherActions, typename PostSmootherActions,
            typename Label = OptionsGroup>
  using solve = tmpl::list<
      async_solvers::PrepareSolve<FieldsTag, OptionsGroup, SourceTag, Label,
                                  Tags::IsFinestGrid, false>,
      detail::ReceiveResidualFromFinerGrid<Dim, FieldsTag, OptionsGroup,
                                           SourceTag>,
      detail::PreparePreSmoothing<FieldsTag, OptionsGroup, SourceTag>,
      PreSmootherActions,
      detail::SkipPostsmoothingAtBottom<FieldsTag, OptionsGroup, SourceTag>,
      detail::SendResidualToCoarserGrid<FieldsTag, OptionsGroup,
                                        ResidualIsMassiveTag, SourceTag>,
      detail::ReceiveCorrectionFromCoarserGrid<Dim, FieldsTag, OptionsGroup,
                                               SourceTag>,
      PostSmootherActions,
      detail::SendCorrectionToFinerGrid<FieldsTag, OptionsGroup, SourceTag>,
      detail::ObserveVolumeData<FieldsTag, OptionsGroup, SourceTag>,
      async_solvers::CompleteStep<FieldsTag, OptionsGroup, SourceTag, Label,
                                  Tags::IsFinestGrid, false>>;
};

}  // namespace LinearSolver::multigrid
