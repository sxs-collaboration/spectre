// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace LinearSolver {
/// Actions applicable to any parallel linear solver
namespace Actions {

/*!
 * \brief Make the iterative linear solve the identity operation on the source
 * vector if no iterations were performed at all. Useful for disabling a
 * preconditioner by setting its number of iterations to zero.
 *
 * When the linear solve is skipped, i.e. when it performs no iterations because
 * its number of iterations is set to zero, this action assumes \f$A=1\f$ so
 * \f$Ax=b\f$ solves to \f$x=b\f$. This is useful when the solver is used as
 * preconditioner, because then we can disable preconditioning by just not
 * iterating the preconditioner, i.e. by setting its number of iterations to
 * zero in the input file.
 *
 * To use this action, insert it into the action list just after iterating the
 * linear solver, i.e. after its `solve` action list:
 *
 * \snippet LinearSolverAlgorithmTestHelpers.hpp make_identity_if_skipped
 *
 * This action will set the `LinearSolverType::fields_tag` to the
 * `LinearSolverType::source_tag` whenever the linear solver has converged with
 * the reason `Convergence::Reason::NumIterations` or
 * `Convergence::Reason::MaxIterations` without actually having performed any
 * iterations.
 *
 * \par Details:
 * The standard behaviour of most linear solvers (i.e. when _not_ using this
 * action) is to keep the fields at their initial guess \f$x=x_0\f$ when it
 * performs no iterations. That behaviour is not handled well when the solver is
 * used as preconditioner and its parent always makes it start at \f$x_0=0\f$.
 * This is a reasonable choice to start the preconditioner but it also means
 * that the preconditioner doesn't reduce to the identity when it performs no
 * iterations. Using this action means not iterating the preconditioner at all
 * essentially disables preconditioning, switching the parent solver to an
 * unpreconditioned solve with some runtime and memory overhead associated with
 * initializing the preconditioner.
 */
template <typename LinearSolverType>
struct MakeIdentityIfSkipped {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& has_converged = get<Convergence::Tags::HasConverged<
        typename LinearSolverType::options_group>>(box);
    if (has_converged and
        (has_converged.reason() == Convergence::Reason::NumIterations or
         has_converged.reason() == Convergence::Reason::MaxIterations) and
        has_converged.num_iterations() == 0) {
      db::mutate<typename LinearSolverType::fields_tag>(
          make_not_null(&box),
          [](const auto fields, const auto& source) noexcept {
            *fields = source;
          },
          get<typename LinearSolverType::source_tag>(box));
    }
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace LinearSolver
