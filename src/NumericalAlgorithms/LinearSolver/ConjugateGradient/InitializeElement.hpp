// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace cg_detail {
template <typename Metavariables>
struct ResidualMonitor;
template <typename BroadcastTarget>
struct InitializeResidual;
}  // namespace cg_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace cg_detail {

template <typename Metavariables, Initialization::MergePolicy MergePolicy =
                                      Initialization::MergePolicy::Error>
struct InitializeElement {
 private:
  using fields_tag = typename Metavariables::system::fields_tag;
  using source_tag = db::add_tag_prefix<::Tags::Source, fields_tag>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DataBox, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags =
        db::AddSimpleTags<residual_tag, LinearSolver::Tags::HasConverged>;
    using compute_tags = db::AddComputeTags<>;

    db::mutate<operand_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<operand_tag>*> p,
           const db::item_type<source_tag> b,
           const db::item_type<operator_applied_to_fields_tag> Ax) noexcept {
          *p = b - Ax;
        },
        get<source_tag>(box), get<operator_applied_to_fields_tag>(box));
    auto residual = db::item_type<residual_tag>{get<operand_tag>(box)};

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    Parallel::contribute_to_reduction<
        cg_detail::InitializeResidual<ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            inner_product(residual, residual)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeElement, simple_tags,
                                             compute_tags, MergePolicy>(
            std::move(box), std::move(residual),
            db::item_type<LinearSolver::Tags::HasConverged>{}));
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
