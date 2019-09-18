// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
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
template <typename Metavariables, typename FieldsTag>
struct ResidualMonitor;
}  // namespace cg_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace cg_detail {

template <typename FieldsTag>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using source_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<LinearSolver::Tags::IterationId, operand_tag, residual_tag>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id,
           const gsl::not_null<db::item_type<operand_tag>*> operand,
           const gsl::not_null<db::item_type<residual_tag>*> residual,
           const db::item_type<source_tag>& source,
           const db::item_type<operator_applied_to_fields_tag>&
               operator_applied_to_fields) noexcept {
          // We have not started iterating yet, so we initialize the current
          // iteration ID such that the _next_ iteration ID is zero.
          *iteration_id = std::numeric_limits<size_t>::max();
          *operand = source - operator_applied_to_fields;
          *residual = *operand;
        },
        get<source_tag>(box), get<operator_applied_to_fields_tag>(box));

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    const auto& residual = get<residual_tag>(box);
    Parallel::contribute_to_reduction<
        cg_detail::InitializeResidual<FieldsTag, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            inner_product(residual, residual)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag>>(cache));

    return {
        std::move(box),
        // Terminate algorithm for now. The `ResidualMonitor` will receive the
        // reduction that is performed above and then broadcast to the following
        // action, which is responsible for restarting the algorithm.
        true};
  }
};

template <typename FieldsTag>
struct InitializeHasConverged {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<LinearSolver::Tags::HasConverged,
                                              DataBox>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const db::item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<LinearSolver::Tags::HasConverged>(
        make_not_null(&box),
        [&has_converged](const gsl::not_null<
                         db::item_type<LinearSolver::Tags::HasConverged>*>
                             local_has_converged) noexcept {
          *local_has_converged = has_converged;
        });

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

struct PrepareStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<LinearSolver::Tags::IterationId>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<LinearSolver::Tags::IterationId>*>
               iteration_id,
           const db::const_item_type<::Tags::Next<
               LinearSolver::Tags::IterationId>>& next_iteration_id) noexcept {
          *iteration_id = next_iteration_id;
        },
        get<::Tags::Next<LinearSolver::Tags::IterationId>>(box));
    return {std::move(box)};
  }
};

template <typename FieldsTag>
struct PerformStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    using fields_tag = FieldsTag;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using operator_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;

    ASSERT(get<LinearSolver::Tags::IterationId>(box) !=
               std::numeric_limits<size_t>::max(),
           "Linear solve iteration ID is at initial state. Did you forget to "
           "invoke 'PrepareStep'?");

    // At this point Ap must have been computed in a previous action
    // We compute the inner product <p,p> w.r.t A. This requires a global
    // reduction.
    const double local_conj_grad_inner_product =
        inner_product(get<operand_tag>(box), get<operator_tag>(box));

    Parallel::contribute_to_reduction<
        ComputeAlpha<FieldsTag, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            local_conj_grad_inner_product},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag>>(cache));

    // Terminate algorithm for now. The `ResidualMonitor` will receive the
    // reduction that is performed above and then broadcast to the following
    // action, which is responsible for restarting the algorithm.
    return {std::move(box), true};
  }
};

template <typename FieldsTag>
struct UpdateFieldValues {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using operator_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<residual_tag, DataBox> and
                     db::tag_is_retrievable_v<fields_tag, DataBox> and
                     db::tag_is_retrievable_v<operand_tag, DataBox> and
                     db::tag_is_retrievable_v<operator_tag, DataBox>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const double alpha) noexcept {
    // Received global reduction result, proceed with conjugate gradient.
    db::mutate<residual_tag, fields_tag>(
        make_not_null(&box),
        [alpha](const gsl::not_null<db::item_type<residual_tag>*> r,
                const gsl::not_null<db::item_type<fields_tag>*> x,
                const db::const_item_type<operand_tag>& p,
                const db::const_item_type<operator_tag>& Ap) noexcept {
          *x += alpha * p;
          *r -= alpha * Ap;
        },
        get<operand_tag>(box), get<operator_tag>(box));

    // Compute new residual norm in a second global reduction
    const auto& r = get<residual_tag>(box);
    const double local_residual_magnitude_square = inner_product(r, r);

    Parallel::contribute_to_reduction<
        UpdateResidual<FieldsTag, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            local_residual_magnitude_square},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag>>(cache));
  }
};

template <typename FieldsTag>
struct UpdateOperand {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<fields_tag, DataBox> and
                     db::tag_is_retrievable_v<LinearSolver::Tags::HasConverged,
                                              DataBox> and
                     db::tag_is_retrievable_v<residual_tag, DataBox>> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const double res_ratio,
                    const db::const_item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<operand_tag, LinearSolver::Tags::HasConverged>(
        make_not_null(&box),
        [
          res_ratio, &has_converged
        ](const gsl::not_null<db::item_type<operand_tag>*> p,
          const gsl::not_null<db::item_type<LinearSolver::Tags::HasConverged>*>
              local_has_converged,
          const db::const_item_type<residual_tag>& r) noexcept {
          *p = r + res_ratio * *p;
          *local_has_converged = has_converged;
        },
        get<residual_tag>(box));

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
