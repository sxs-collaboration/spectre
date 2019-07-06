// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitorActions.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace cg_detail {
template <typename Metavariables>
struct ResidualMonitor;
}  // namespace cg_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace cg_detail {

struct InitializeHasConverged {
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<db::tag_is_retrievable_v<
                   typename Metavariables::system::fields_tag, DataBox> and
               db::tag_is_retrievable_v<LinearSolver::Tags::HasConverged,
                                        DataBox>> = nullptr>
  static void apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const db::item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<LinearSolver::Tags::HasConverged>(
        make_not_null(&box), [&has_converged](
                                 const gsl::not_null<db::item_type<
                                     LinearSolver::Tags::HasConverged>*>
                                     local_has_converged) noexcept {
          *local_has_converged = has_converged;
        });
  }
};

struct PerformStep {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<DataBox&&, bool> apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using operator_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;

    // At this point Ap must have been computed in a previous action
    // We compute the inner product <p,p> w.r.t A. This requires a global
    // reduction.
    const double local_conj_grad_inner_product =
        inner_product(get<operand_tag>(box), get<operator_tag>(box));

    Parallel::contribute_to_reduction<ComputeAlpha<ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            local_conj_grad_inner_product},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    // Terminate algorithm for now. The reduction will be broadcast to the
    // next action which is responsible for restarting the algorithm.
    return {std::move(box), true};
  }
};

struct UpdateFieldValues {
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<db::tag_is_retrievable_v<
                   typename Metavariables::system::fields_tag, DataBox> and
               db::tag_is_retrievable_v<LinearSolver::Tags::HasConverged,
                                        DataBox>> = nullptr>
  static auto apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const double alpha) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using operator_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;
    using residual_tag =
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

    // Received global reduction result, proceed with conjugate gradient.
    db::mutate<residual_tag, fields_tag>(
        make_not_null(&box),
        [alpha](const gsl::not_null<db::item_type<residual_tag>*> r,
                const gsl::not_null<db::item_type<fields_tag>*> x,
                const db::item_type<operand_tag>& p,
                const db::item_type<operator_tag>& Ap) noexcept {
          *x += alpha * p;
          *r -= alpha * Ap;
        },
        get<operand_tag>(box), get<operator_tag>(box));

    // Compute new residual norm in a second global reduction
    const auto& r = get<residual_tag>(box);
    const double local_residual_magnitude_square = inner_product(r, r);

    Parallel::contribute_to_reduction<UpdateResidual<ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            local_residual_magnitude_square},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));
  }
};

struct UpdateOperand {
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<db::tag_is_retrievable_v<
                   typename Metavariables::system::fields_tag, DataBox> and
               db::tag_is_retrievable_v<LinearSolver::Tags::HasConverged,
                                        DataBox>> = nullptr>
  static auto apply(DataBox& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const double res_ratio,
                    const db::item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using residual_tag =
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

    // Prepare conjugate gradient for next iteration
    db::mutate<operand_tag, LinearSolver::Tags::HasConverged,
               LinearSolver::Tags::IterationId>(
        make_not_null(&box),
        [
          res_ratio, &has_converged
        ](const gsl::not_null<db::item_type<operand_tag>*> p,
          const gsl::not_null<db::item_type<LinearSolver::Tags::HasConverged>*>
              local_has_converged,
          const gsl::not_null<db::item_type<LinearSolver::Tags::IterationId>*>
              iteration_id,
          const db::item_type<residual_tag>& r) noexcept {
          *p = r + res_ratio * *p;
          *local_has_converged = has_converged;
          (*iteration_id)++;
        },
        get<residual_tag>(box));

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
