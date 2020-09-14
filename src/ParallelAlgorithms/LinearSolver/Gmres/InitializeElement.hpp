// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace LinearSolver::gmres::detail {

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using initial_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, fields_tag>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using preconditioned_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;
  using operator_applied_to_operand_tag = db::add_tag_prefix<
      LinearSolver::Tags::OperatorAppliedTo,
      std::conditional_t<Preconditioned, preconditioned_operand_tag,
                         operand_tag>>;
  using orthogonalization_iteration_id_tag =
      LinearSolver::Tags::Orthogonalization<
          LinearSolver::Tags::IterationId<OptionsGroup>>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto initial_box = ::Initialization::merge_into_databox<
        InitializeElement,
        db::AddSimpleTags<LinearSolver::Tags::IterationId<OptionsGroup>,
                          initial_fields_tag, operator_applied_to_fields_tag,
                          operand_tag, operator_applied_to_operand_tag,
                          orthogonalization_iteration_id_tag, basis_history_tag,
                          LinearSolver::Tags::HasConverged<OptionsGroup>>>(
        std::move(box),
        // The `PrepareSolve` action populates these tags with initial values,
        // except for `operator_applied_to_fields_tag` which is expected to be
        // filled at that point and `operator_applied_to_operand_tag` which is
        // expected to be updated in every iteration of the algorithm.
        std::numeric_limits<size_t>::max(), typename initial_fields_tag::type{},
        typename operator_applied_to_fields_tag::type{},
        typename operand_tag::type{},
        typename operator_applied_to_operand_tag::type{},
        std::numeric_limits<size_t>::max(), typename basis_history_tag::type{},
        Convergence::HasConverged{});

    if constexpr (not Preconditioned) {
      return std::make_tuple(std::move(initial_box));
    } else {
      using preconditioned_basis_history_tag =
          LinearSolver::Tags::KrylovSubspaceBasis<preconditioned_operand_tag>;

      return std::make_tuple(::Initialization::merge_into_databox<
                             InitializeElement,
                             db::AddSimpleTags<preconditioned_basis_history_tag,
                                               preconditioned_operand_tag>>(
          std::move(initial_box),
          typename preconditioned_basis_history_tag::type{},
          typename preconditioned_operand_tag::type{}));
    }
  }
};

}  // namespace LinearSolver::gmres::detail
