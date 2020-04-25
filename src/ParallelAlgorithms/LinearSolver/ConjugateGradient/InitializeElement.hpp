// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace LinearSolver {
namespace cg_detail {

template <typename FieldsTag, typename OptionsGroup>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using operator_applied_to_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        ::Tags::NextCompute<LinearSolver::Tags::IterationId<OptionsGroup>>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<LinearSolver::Tags::IterationId<OptionsGroup>,
                              operator_applied_to_fields_tag, operand_tag,
                              operator_applied_to_operand_tag, residual_tag,
                              LinearSolver::Tags::HasConverged<OptionsGroup>>,
            compute_tags>(
            std::move(box),
            // The `PrepareSolve` action populates these tags with initial
            // values, except for `operator_applied_to_fields_tag` which is
            // expected to be filled at that point and
            // `operator_applied_to_operand_tag` which is expected to be updated
            // in every iteration of the algorithm.
            std::numeric_limits<size_t>::max(),
            db::item_type<operator_applied_to_fields_tag>{},
            db::item_type<operand_tag>{},
            db::item_type<operator_applied_to_operand_tag>{},
            db::item_type<residual_tag>{}, Convergence::HasConverged{}));
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
