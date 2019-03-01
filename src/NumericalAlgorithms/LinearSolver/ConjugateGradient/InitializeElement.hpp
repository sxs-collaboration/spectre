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

template <typename Metavariables>
struct InitializeElement {
 private:
  using fields_tag = typename Metavariables::system::fields_tag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  using simple_tags =
      db::AddSimpleTags<LinearSolver::Tags::IterationId,
                        ::Tags::Next<LinearSolver::Tags::IterationId>,
                        residual_tag, LinearSolver::Tags::HasConverged>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList, typename ArrayIndex, typename ParallelComponent>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ParallelComponent* const /*meta*/,
      const db::item_type<db::add_tag_prefix<::Tags::Source, fields_tag>>& b,
      const db::item_type<db::add_tag_prefix<
          LinearSolver::Tags::OperatorAppliedTo, fields_tag>>& Ax) noexcept {
    db::mutate<operand_tag>(make_not_null(&box), [
      &b, &Ax
    ](const gsl::not_null<db::item_type<operand_tag>*> p) noexcept {
      *p = b - Ax;
    });
    auto r = db::item_type<residual_tag>(get<operand_tag>(box));

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    Parallel::contribute_to_reduction<
        cg_detail::InitializeResidual<ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            inner_product(r, r)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), db::item_type<LinearSolver::Tags::IterationId>{0},
        db::item_type<::Tags::Next<LinearSolver::Tags::IterationId>>{1},
        std::move(r), db::item_type<LinearSolver::Tags::HasConverged>{});
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
