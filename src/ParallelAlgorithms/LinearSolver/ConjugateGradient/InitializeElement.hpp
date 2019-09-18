// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
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

template <typename FieldsTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
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
        ::Tags::NextCompute<LinearSolver::Tags::IterationId>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<LinearSolver::Tags::IterationId, residual_tag,
                              LinearSolver::Tags::HasConverged>,
            compute_tags>(std::move(box),
                          // The `PrepareSolve` action populates these tags with
                          // initial values
                          std::numeric_limits<size_t>::max(),
                          db::item_type<residual_tag>{},
                          db::item_type<LinearSolver::Tags::HasConverged>{}));
  }
};

}  // namespace cg_detail
}  // namespace LinearSolver
