// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
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
namespace gmres_detail {
template <typename>
struct ResidualMonitor;
template <typename>
struct InitializeResidualMagnitude;
struct InitializeSourceMagnitude;
}  // namespace gmres_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace gmres_detail {

template <typename Metavariables>
struct InitializeElement {
 private:
  using fields_tag = typename Metavariables::system::fields_tag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using operator_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;
  using orthogonalization_iteration_id_tag =
      db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                         LinearSolver::Tags::IterationId>;
  using basis_history_tag = LinearSolver::Tags::KrylovSubspaceBasis<fields_tag>;

 public:
  using simple_tags =
      db::AddSimpleTags<LinearSolver::Tags::IterationId,
                        ::Tags::Next<LinearSolver::Tags::IterationId>,
                        operand_tag, operator_tag,
                        orthogonalization_iteration_id_tag, basis_history_tag>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList, typename ArrayIndex, typename ParallelComponent>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ParallelComponent* const /*meta*/,
      const db::item_type<db::add_tag_prefix<::Tags::Source, fields_tag>>& b,
      const db::item_type<db::add_tag_prefix<
          LinearSolver::Tags::OperatorAppliedTo, fields_tag>>& Ax) noexcept {
    Parallel::contribute_to_reduction<gmres_detail::InitializeSourceMagnitude>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            inner_product(b, b)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    db::item_type<operand_tag> q = b - Ax;

    Parallel::contribute_to_reduction<
        gmres_detail::InitializeResidualMagnitude<ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            inner_product(q, q)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    auto Aq = make_with_value<db::item_type<operator_tag>>(
        q, std::numeric_limits<double>::signaling_NaN());
    db::item_type<basis_history_tag> basis_history{};

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), IterationId{0}, IterationId{1}, std::move(q),
        std::move(Aq), IterationId{0}, std::move(basis_history));
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
