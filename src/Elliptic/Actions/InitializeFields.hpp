// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::Actions {

/*!
 * \brief Initialize the dynamic fields of the elliptic system, i.e. those we
 * solve for.
 *
 * Uses:
 * - System:
 *   - `fields_tag`
 * - DataBox:
 *   - `InitialGuessTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename System, typename InitialGuessTag>
struct InitializeFields {
 private:
  using system = System;
  using fields_tag = typename system::fields_tag;

 public:
  using simple_tags = tmpl::list<
      fields_tag,
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& initial_guess = db::get<InitialGuessTag>(box);
    auto initial_fields = variables_from_tagged_tuple(initial_guess.variables(
        inertial_coords, typename fields_tag::tags_list{}));
    ::Initialization::mutate_assign<tmpl::list<fields_tag>>(
        make_not_null(&box), std::move(initial_fields));
    // For now we assume the initial data is zero so we don't need to apply the
    // DG operator but may just set it to zero as well. This will be replaced
    // ASAP by applying the DG operator to the initial fields.
    db::mutate<
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>>(
        make_not_null(&box),
        [&inertial_coords](
            const auto linear_operator_applied_to_fields) noexcept {
          linear_operator_applied_to_fields->initialize(
              inertial_coords.begin()->size(), 0.);
        });
    return {std::move(box)};
  }
};

}  // namespace elliptic::Actions
