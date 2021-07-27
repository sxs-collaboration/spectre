// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t Dim>
struct ElementId;
/// \endcond

namespace LinearSolver::multigrid {

template <size_t Dim, typename ReceiveTags>
struct DataFromChildrenInboxTag
    : public Parallel::InboxInserters::Map<
          DataFromChildrenInboxTag<Dim, ReceiveTags>> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id,
               FixedHashMap<two_to_the(Dim), ElementId<Dim>,
                            tuples::tagged_tuple_from_typelist<ReceiveTags>,
                            boost::hash<ElementId<Dim>>>>;
};

/// Actions related to the Multigrid linear solver
namespace Actions {
/*!
 * \brief Communicate and project the `FieldsTags` to the next-coarser grid
 * in the multigrid hierarchy
 *
 * \tparam FieldsTags These tags will be communicated and projected. They can
 * hold any type that works with `::apply_matrices` and supports addition, e.g.
 * `Variables`.
 * \tparam OptionsGroup The option group identifying the multigrid solver
 * \tparam FieldsAreMassiveTag A boolean tag in the DataBox that indicates
 * whether or not the `FieldsTags` have already been multiplied by the mass
 * matrix. This setting influences the way the fields are projected. In
 * particular, the mass matrix already includes a Jacobian factor, so the
 * difference in size between the parent and the child element is already
 * accounted for.
 * \tparam ReceiveTags The projected fields will be stored in these tags
 * (default: `FieldsTags`).
 */
template <typename FieldsTags, typename OptionsGroup,
          typename FieldsAreMassiveTag, typename ReceiveTags = FieldsTags>
struct SendFieldsToCoarserGrid;

/// \cond
template <typename... FieldsTags, typename OptionsGroup,
          typename FieldsAreMassiveTag, typename... ReceiveTags>
struct SendFieldsToCoarserGrid<tmpl::list<FieldsTags...>, OptionsGroup,
                               FieldsAreMassiveTag,
                               tmpl::list<ReceiveTags...>> {
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Skip restriction on coarsest level
    const auto& parent_id = db::get<Tags::ParentId<Dim>>(box);
    if (not parent_id.has_value()) {
      return {std::move(box)};
    }

    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Send fields to coarser grid\n", element_id,
                       Options::name<OptionsGroup>(), iteration_id);
    }

    // Restrict the fields to the coarser (parent) grid.
    // We restrict before sending the data so the restriction operation is
    // parellelized. The parent only needs to sum up all child contributions.
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& parent_mesh = db::get<Tags::ParentMesh<Dim>>(box);
    ASSERT(
        parent_mesh.has_value(),
        "Should have a parent mesh, because a parent ID is set. This element: "
            << element_id << ", parent element: " << *parent_id);
    const auto child_size =
        domain::child_size(element_id.segment_ids(), parent_id->segment_ids());
    bool massive = false;
    if constexpr (not std::is_same_v<FieldsAreMassiveTag, void>) {
      massive = db::get<FieldsAreMassiveTag>(box);
    }
    tuples::TaggedTuple<ReceiveTags...> restricted_fields{};
    if (Spectral::needs_projection(mesh, *parent_mesh, child_size)) {
      const auto restriction_operator =
          Spectral::projection_matrix_child_to_parent(mesh, *parent_mesh,
                                                      child_size, massive);
      const auto restrict_fields = [&restricted_fields, &restriction_operator,
                                    &mesh](const auto receive_tag_v,
                                           const auto& fields) noexcept {
        using receive_tag = std::decay_t<decltype(receive_tag_v)>;
        get<receive_tag>(restricted_fields) = typename receive_tag::type(
            apply_matrices(restriction_operator, fields, mesh.extents()));
        return '0';
      };
      expand_pack(restrict_fields(ReceiveTags{}, db::get<FieldsTags>(box))...);
    } else {
      expand_pack(
          (get<ReceiveTags>(restricted_fields) =
               typename ReceiveTags::type(db::get<FieldsTags>(box)))...);
    }

    // Send restricted fields to the parent
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    Parallel::receive_data<
        DataFromChildrenInboxTag<Dim, tmpl::list<ReceiveTags...>>>(
        receiver_proxy[*parent_id], iteration_id,
        std::make_pair(element_id, std::move(restricted_fields)));
    return {std::move(box)};
  }
};
/// \endcond

/// Receive the `FieldsTags` communicated from the finer grid in the multigrid
/// hierarchy.
///
/// \see LinearSolver::multigrid::Actions::SendFieldsToCoarserGrid
template <size_t Dim, typename FieldsTags, typename OptionsGroup,
          typename ReceiveTags = FieldsTags>
struct ReceiveFieldsFromFinerGrid;

/// \cond
template <size_t Dim, typename FieldsTags, typename OptionsGroup,
          typename... ReceiveTags>
struct ReceiveFieldsFromFinerGrid<Dim, FieldsTags, OptionsGroup,
                                  tmpl::list<ReceiveTags...>> {
  using inbox_tags =
      tmpl::list<DataFromChildrenInboxTag<Dim, tmpl::list<ReceiveTags...>>>;
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<Dim>& element_id, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    // Skip on finest grid
    const auto& child_ids = db::get<Tags::ChildIds<Dim>>(box);
    if (child_ids.empty()) {
      return {std::move(box), Parallel::AlgorithmExecution::Continue};
    }

    // Wait for data from finer grid
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    auto& inbox =
        tuples::get<DataFromChildrenInboxTag<Dim, tmpl::list<ReceiveTags...>>>(
            inboxes);
    const auto received_this_iteration = inbox.find(iteration_id);
    if (received_this_iteration == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }
    const auto& received_children_data = received_this_iteration->second;
    for (const auto& child_id : child_ids) {
      if (received_children_data.find(child_id) ==
          received_children_data.end()) {
        return {std::move(box), Parallel::AlgorithmExecution::Retry};
      }
    }
    auto children_data = std::move(inbox.extract(iteration_id).mapped());

    if (UNLIKELY(db::get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Receive fields from finer grid\n",
                       element_id, Options::name<OptionsGroup>(), iteration_id);
    }

    // Assemble restricted data from children
    const auto assemble_children_data =
        [&children_data](const auto source, const auto receive_tag_v) noexcept {
          using receive_tag = std::decay_t<decltype(receive_tag_v)>;
          // Move the first child data directly into the buffer, then add the
          // data from the remaining children.
          auto child_id_and_data = children_data.begin();
          *source = std::move(get<receive_tag>(child_id_and_data->second));
          ++child_id_and_data;
          while (child_id_and_data != children_data.end()) {
            *source += get<receive_tag>(child_id_and_data->second);
            ++child_id_and_data;
          }
          return '0';
        };
    expand_pack(db::mutate<ReceiveTags>(
        make_not_null(&box), assemble_children_data, ReceiveTags{})...);

    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};
/// \endcond

}  // namespace Actions
}  // namespace LinearSolver::multigrid
