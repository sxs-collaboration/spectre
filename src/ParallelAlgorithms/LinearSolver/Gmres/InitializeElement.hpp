// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
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
namespace gmres_detail {

template <typename FieldsTag, typename OptionsGroup>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using initial_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::Initial, fields_tag>;
  using orthogonalization_iteration_id_tag =
      db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                         LinearSolver::Tags::IterationId<OptionsGroup>>;
  using basis_history_tag = LinearSolver::Tags::KrylovSubspaceBasis<fields_tag>;

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
                              initial_fields_tag,
                              orthogonalization_iteration_id_tag,
                              basis_history_tag,
                              LinearSolver::Tags::HasConverged<OptionsGroup>>,
            compute_tags>(std::move(box),
                          // The `PrepareSolve` action populates these tags with
                          // initial values
                          std::numeric_limits<size_t>::max(),
                          db::item_type<initial_fields_tag>{},
                          std::numeric_limits<size_t>::max(),
                          db::item_type<basis_history_tag>{},
                          Convergence::HasConverged{}));
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
