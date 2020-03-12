// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/InnerProduct.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace cg_detail {
template <typename Metavariables, typename FieldsTag>
struct ResidualMonitor;
template <typename FieldsTag, typename BroadcastTarget>
struct InitializeResidual;
}  // namespace cg_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace cg_detail {

template <typename FieldsTag, Initialization::MergePolicy MergePolicy =
                                  Initialization::MergePolicy::Error>
struct InitializeElement {
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
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<operand_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<operand_tag>*> operand,
           const db::const_item_type<source_tag>& source,
           const db::const_item_type<operator_applied_to_fields_tag>&
               operator_applied_to_fields) noexcept {
          *operand = source - operator_applied_to_fields;
        },
        get<source_tag>(box), get<operator_applied_to_fields_tag>(box));
    auto residual = db::item_type<residual_tag>{get<operand_tag>(box)};

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    Parallel::contribute_to_reduction<
        cg_detail::InitializeResidual<FieldsTag, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            inner_product(residual, residual)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag>>(cache));

    using compute_tags = db::AddComputeTags<
        ::Tags::NextCompute<LinearSolver::Tags::IterationId>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<LinearSolver::Tags::IterationId, residual_tag,
                              LinearSolver::Tags::HasConverged>,
            compute_tags, MergePolicy>(
            std::move(box),
            // We have not started iterating yet, so we initialize the current
            // iteration ID such that the _next_ iteration ID is zero.
            std::numeric_limits<size_t>::max(), std::move(residual),
            db::item_type<LinearSolver::Tags::HasConverged>{}),
        // Terminate algorithm for now. The `ResidualMonitor` will receive the
        // reduction that is performed above and then broadcast to the following
        // action, which is responsible for restarting the algorithm.
        true);
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
                    const db::const_item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<LinearSolver::Tags::HasConverged>(
        make_not_null(&box), [&has_converged](
                                 const gsl::not_null<db::item_type<
                                     LinearSolver::Tags::HasConverged>*>
                                     local_has_converged) noexcept {
          *local_has_converged = has_converged;
        });

    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
