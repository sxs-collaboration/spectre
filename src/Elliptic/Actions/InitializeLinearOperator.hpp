// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

struct InitializeLinearOperator {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using linear_operator_applied_to_fields_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();

    // Since the initial data is zero we don't need to apply the DG operator but
    // may just set it to zero as well. Once this condition is relaxed we will
    // have to add a communication step to the initialization that computes the
    // DG operator applied to the initial data.
    db::mutate<linear_operator_applied_to_fields_tag>(
        make_not_null(&box),
        [&num_grid_points](
            const gsl::not_null<
                db::item_type<linear_operator_applied_to_fields_tag>*>
                linear_operator_applied_to_fields) noexcept {
          *linear_operator_applied_to_fields =
              db::item_type<linear_operator_applied_to_fields_tag>{
                  num_grid_points, 0.};
        });

    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace elliptic
