// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/LinearSolver/Observe.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace cg_detail {
struct InitializeHasConverged;
struct UpdateFieldValues;
struct UpdateOperand;
}  // namespace cg_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace cg_detail {

template <typename BroadcastTarget>
struct InitializeResidual {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double residual_square) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using residual_square_tag = db::add_tag_prefix<
        LinearSolver::Tags::MagnitudeSquare,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
    using residual_magnitude_tag = db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
    using initial_residual_magnitude_tag =
        db::add_tag_prefix<LinearSolver::Tags::Initial, residual_magnitude_tag>;

    db::mutate<residual_square_tag>(
        make_not_null(&box), [residual_square](
                                 const gsl::not_null<double*>
                                     local_residual_square) noexcept {
          *local_residual_square = residual_square;
        });
    // Perform a separate `db::mutate` so that we can retrieve the
    // `residual_magnitude_tag` from the compute item
    db::mutate<initial_residual_magnitude_tag>(
        make_not_null(&box),
        [](const gsl::not_null<double*> local_initial_residual_magnitude,
           const double& initial_residual_magnitude) noexcept {
          *local_initial_residual_magnitude = initial_residual_magnitude;
        },
        get<residual_magnitude_tag>(box));

    LinearSolver::observe_detail::contribute_to_reduction_observer(box, cache);

    // Determine whether the linear solver has converged. This invokes the
    // compute item.
    const auto& has_converged = db::get<LinearSolver::Tags::HasConverged>(box);

    if (UNLIKELY(has_converged and
                 static_cast<int>(get<::Tags::Verbosity>(box)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf(
          "The linear solver has converged without any iterations: %s",
          has_converged);
    }

    Parallel::simple_action<InitializeHasConverged>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        has_converged);
  }
};

template <typename BroadcastTarget>
struct ComputeAlpha {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double conj_grad_inner_product) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using residual_square_tag = db::add_tag_prefix<
        LinearSolver::Tags::MagnitudeSquare,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;

    Parallel::simple_action<UpdateFieldValues>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        get<residual_square_tag>(box) / conj_grad_inner_product);
  }
};

template <typename BroadcastTarget>
struct UpdateResidual {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double residual_square) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using residual_square_tag = db::add_tag_prefix<
        LinearSolver::Tags::MagnitudeSquare,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
    using residual_magnitude_tag = db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;

    // Compute the residual ratio before mutating the DataBox
    const double res_ratio = residual_square / get<residual_square_tag>(box);

    db::mutate<residual_square_tag, LinearSolver::Tags::IterationId>(
        make_not_null(&box),
        [residual_square](
            const gsl::not_null<double*> local_residual_square,
            const gsl::not_null<db::item_type<LinearSolver::Tags::IterationId>*>
                iteration_id) noexcept {
          *local_residual_square = residual_square;
          // Prepare for the next iteration
          (*iteration_id)++;
        });

    // At this point, the iteration is complete. We proceed with observing,
    // logging and checking convergence before broadcasting back to the
    // elements.

    LinearSolver::observe_detail::contribute_to_reduction_observer(box, cache);

    // Determine whether the linear solver has converged. This invokes the
    // compute item.
    const auto& has_converged = get<LinearSolver::Tags::HasConverged>(box);

    // Do some logging
    if (UNLIKELY(static_cast<int>(get<::Tags::Verbosity>(box)) >=
                 static_cast<int>(::Verbosity::Verbose))) {
      Parallel::printf(
          "Linear solver iteration %zu done. Remaining residual: %e\n",
          get<LinearSolver::Tags::IterationId>(box),
          get<residual_magnitude_tag>(box));
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(get<::Tags::Verbosity>(box)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf("The linear solver has converged in %zu iterations: %s",
                       get<LinearSolver::Tags::IterationId>(box),
                       has_converged);
    }

    Parallel::simple_action<UpdateOperand>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), res_ratio,
        has_converged);
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
