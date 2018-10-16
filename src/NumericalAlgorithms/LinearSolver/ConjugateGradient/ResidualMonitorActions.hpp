// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace cg_detail {
struct UpdateFieldValues;
struct UpdateOperand;
}  // namespace cg_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace cg_detail {

struct InitializeResidual {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                db::add_tag_prefix<LinearSolver::Tags::ResidualMagnitudeSquare,
                                   typename Metavariables::system::fields_tag>,
                DbTags>...>> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double res_new) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using residual_square_tag =
        db::add_tag_prefix<LinearSolver::Tags::ResidualMagnitudeSquare,
                           fields_tag>;

    db::mutate<residual_square_tag>(
        make_not_null(&box), [res_new](const gsl::not_null<double*>
                                           res_old) noexcept {
          *res_old = res_new;
        });
  }
};

template <typename BroadcastTarget>
struct ComputeAlpha {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                db::add_tag_prefix<LinearSolver::Tags::ResidualMagnitudeSquare,
                                   typename Metavariables::system::fields_tag>,
                DbTags>...>> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double conj_grad_inner_product) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using residual_square_tag =
        db::add_tag_prefix<LinearSolver::Tags::ResidualMagnitudeSquare,
                           fields_tag>;

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
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                db::add_tag_prefix<LinearSolver::Tags::ResidualMagnitudeSquare,
                                   typename Metavariables::system::fields_tag>,
                DbTags>...>> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double res_new) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using residual_square_tag =
        db::add_tag_prefix<LinearSolver::Tags::ResidualMagnitudeSquare,
                           fields_tag>;

    const double residual = sqrt(res_new);
    const double res_ratio = res_new / get<residual_square_tag>(box);

    if (static_cast<int>(get<::Tags::Verbosity>(box)) >=
        static_cast<int>(::Verbosity::Verbose)) {
      Parallel::printf(
          "Linear solver iteration %d done. Remaining residual: %e\n",
          get<LinearSolver::Tags::IterationId>(box).step_number + 1, residual);
    }

    db::mutate<residual_square_tag, LinearSolver::Tags::IterationId>(
        make_not_null(&box), [res_new](const gsl::not_null<double*> res_old,
                                       const gsl::not_null<IterationId*>
                                           iteration_id) noexcept {
          *res_old = res_new;
          iteration_id->step_number++;
        });

    Parallel::simple_action<UpdateOperand>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), res_ratio,
        equal_within_roundoff(residual, 0.));
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
