// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename...>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace LinearSolver::Schwarz::Actions {

/*!
 * \brief Reset the subdomain solver, clearing its caches related to the linear
 * operator it has solved so far.
 *
 * Invoke this action when the linear operator has changed. For example, an
 * operator representing the linearization of a nonlinear operator changes
 * whenever the point around which it is being linearized gets updated, i.e. in
 * every iteration of a nonlinear solve. Note that the operator does _not_
 * change between iterations of a linear solve, so make sure you place this
 * action _outside_ the looping action list for the linear solve.
 *
 * \par Skipping the reset:
 * Depending on the subdomain solver in use, the reset may incur significant
 * re-initialization cost the next time the subdomain solver is invoked. See the
 * `LinearSolver::Serial::ExplicitInverse` solver for an example of a subdomain
 * solver with high initialization cost. For this reason the reset can be
 * skipped with the option
 * `LinearSolver::Schwarz::Tags::SkipSubdomainSolverResets`. Skipping resets
 * means that caches built up for the linear operator are never cleared, so
 * expensive re-initializations are avoided but the subdomain solves may be
 * increasingly inaccurate or slow down as the linear operator changes over
 * nonlinear solver iterations. Whether or not skipping resets helps with the
 * overall convergence of the solve is highly problem-dependent. A possible
 * optimization would be to decide at runtime whether or not to reset the
 * subdomain solver.
 */
template <typename OptionsGroup>
struct ResetSubdomainSolver {
  using const_global_cache_tags = tmpl::list<
      LinearSolver::Schwarz::Tags::SkipSubdomainSolverResets<OptionsGroup>,
      logging::Tags::Verbosity<OptionsGroup>>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (not get<LinearSolver::Schwarz::Tags::SkipSubdomainSolverResets<
            OptionsGroup>>(box)) {
      if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                   ::Verbosity::Debug)) {
        Parallel::printf("%s %s: Reset subdomain solver\n", element_id,
                         Options::name<OptionsGroup>());
      }
      db::mutate<
          LinearSolver::Schwarz::Tags::SubdomainSolverBase<OptionsGroup>>(
          make_not_null(&box), [](const auto subdomain_solver) {
            // Dereference the gsl::not_null pointer, and then the
            // std::unique_ptr for the subdomain solver's abstract superclass.
            // This needs adjustment if the subdomain solver is stored in the
            // DataBox directly as a derived class and thus there's no
            // std::unique_ptr. Note that std::unique_ptr also has a `reset`
            // function, which must not be confused with the serial linear
            // solver's `reset` function here.
            (*subdomain_solver)->reset();
          });
    }
    return {std::move(box)};
  }
};

}  // namespace LinearSolver::Schwarz::Actions
