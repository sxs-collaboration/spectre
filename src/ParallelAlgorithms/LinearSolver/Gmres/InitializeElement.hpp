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
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
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
  using initial_fields_tag = db::add_tag_prefix<::Tags::Initial, fields_tag>;
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
          Convergence::Tags::IterationId<OptionsGroup>>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;
  using preconditioned_basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<preconditioned_operand_tag>;

 public:
  using simple_tags = tmpl::append<
      tmpl::list<Convergence::Tags::IterationId<OptionsGroup>,
                 initial_fields_tag, operator_applied_to_fields_tag,
                 operand_tag, operator_applied_to_operand_tag,
                 orthogonalization_iteration_id_tag, basis_history_tag,
                 Convergence::Tags::HasConverged<OptionsGroup>>,
      tmpl::conditional_t<Preconditioned,
                          tmpl::list<preconditioned_basis_history_tag,
                                     preconditioned_operand_tag>,
                          tmpl::list<>>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    Initialization::mutate_assign<
        tmpl::list<Convergence::Tags::IterationId<OptionsGroup>,
                   orthogonalization_iteration_id_tag>>(
        make_not_null(&box), std::numeric_limits<size_t>::max(),
        std::numeric_limits<size_t>::max());
    return std::make_tuple(std::move(box));
  }
};

}  // namespace LinearSolver::gmres::detail
