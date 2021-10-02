// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Actions for communicating data on regions that overlap with the subdomains
/// of other elements

#pragma once

#include <cstddef>
#include <map>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t Dim>
struct ElementId;
/// \endcond

namespace LinearSolver::Schwarz {
/// Actions related to the Schwarz solver
namespace Actions {

namespace detail {
template <size_t Dim, typename OverlapFields, typename OptionsGroup>
struct OverlapFieldsTag
    : public Parallel::InboxInserters::Map<
          OverlapFieldsTag<Dim, OverlapFields, OptionsGroup>> {
  using temporal_id = size_t;
  using type = std::map<
      temporal_id,
      OverlapMap<Dim, tmpl::conditional_t<
                          (tmpl::size<OverlapFields>::value > 1),
                          tuples::tagged_tuple_from_typelist<OverlapFields>,
                          typename tmpl::front<OverlapFields>::type>>>;
};
}  // namespace detail

/*!
 * \brief Send data on regions that overlap with other subdomains to their
 * corresponding elements
 *
 * Collect the `OverlapFields` on "intruding overlaps", i.e. regions that
 * overlap with the subdomains of other elements, and send the data to those
 * elements. The `OverlapFields` can be tags holding either `Variables` or
 * `Tensor`s. The `RestrictToOverlap` flag controls whether the tags are simply
 * retrieved from the element and sent as-is (`false`) or only the data that
 * intersect the overlap region are sent (`true`). If `RestrictToOverlap` is
 * `false` this action can also be used to communicate non-tensor data.
 *
 * This actions should be followed by
 * `LinearSolver::Schwarz::Actions::ReceiveOverlapFields` in the action list.
 */
template <typename OverlapFields, typename OptionsGroup, bool RestrictToOverlap>
struct SendOverlapFields;

/// \cond
template <typename... OverlapFields, typename OptionsGroup,
          bool RestrictToOverlap>
struct SendOverlapFields<tmpl::list<OverlapFields...>, OptionsGroup,
                         RestrictToOverlap> {
  using const_global_cache_tags =
      tmpl::list<Tags::MaxOverlap<OptionsGroup>,
                 logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& element = get<domain::Tags::Element<Dim>>(box);

    // Skip communicating if the overlap is empty
    if (UNLIKELY(db::get<Tags::MaxOverlap<OptionsGroup>>(box) == 0 or
                 element.number_of_neighbors() == 0)) {
      return {std::move(box)};
    }

    // Do some logging
    const auto& iteration_id =
        get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Send overlap fields\n", element_id,
                       Options::name<OptionsGroup>(), iteration_id);
    }

    // Send data on intruding overlaps to the corresponding neighbors
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const auto& neighbors = direction_and_neighbors.second;
      // Collect the data on intruding overlaps
      tuples::TaggedTuple<OverlapFields...> overlap_fields{};
      if constexpr (RestrictToOverlap) {
        const auto& element_extents =
            get<domain::Tags::Mesh<Dim>>(box).extents();
        const size_t intruding_extent =
            gsl::at(get<Tags::IntrudingExtents<Dim, OptionsGroup>>(box),
                    direction.dimension());
        expand_pack((get<OverlapFields>(overlap_fields) =
                         LinearSolver::Schwarz::data_on_overlap(
                             db::get<OverlapFields>(box), element_extents,
                             intruding_extent, direction))...);
      } else {
        expand_pack((get<OverlapFields>(overlap_fields) =
                         db::get<OverlapFields>(box))...);
      }
      // We elide the tagged tuple in the inbox tag if only a single tag is
      // communicated. This optimization allows moving the overlap-map into the
      // DataBox in one piece.
      auto& collapsed_overlap_fields = [&overlap_fields]() -> auto& {
        if constexpr (sizeof...(OverlapFields) > 1) {
          return overlap_fields;
        } else {
          return get<OverlapFields...>(overlap_fields);
        }
      }
      ();
      // Copy data to send to neighbors, but move it for the last one
      const auto direction_from_neighbor =
          neighbors.orientation()(direction.opposite());
      for (auto neighbor = neighbors.begin(); neighbor != neighbors.end();
           ++neighbor) {
        Parallel::receive_data<detail::OverlapFieldsTag<
            Dim, tmpl::list<OverlapFields...>, OptionsGroup>>(
            receiver_proxy[*neighbor], iteration_id,
            std::make_pair(
                OverlapId<Dim>{direction_from_neighbor, element.id()},
                (std::next(neighbor) == neighbors.end())
                    // NOLINTNEXTLINE(bugprone-use-after-move)
                    ? std::move(collapsed_overlap_fields)
                    : collapsed_overlap_fields));
      }
    }
    return {std::move(box)};
  }
};
/// \endcond

/*!
 * \brief Receive data from regions of this element's subdomain that overlap
 * with other elements
 *
 * This action waits until overlap data from all neighboring elements has been
 * received and then moves the data into the DataBox as
 * `LinearSolver::Schwarz::Tags::Overlaps<OverlapFields...>`.
 *
 * This actions should be preceded by
 * `LinearSolver::Schwarz::Actions::SendOverlapFields` in the action list.
 */
template <size_t Dim, typename OverlapFields, typename OptionsGroup>
struct ReceiveOverlapFields;

/// \cond
template <size_t Dim, typename... OverlapFields, typename OptionsGroup>
struct ReceiveOverlapFields<Dim, tmpl::list<OverlapFields...>, OptionsGroup> {
 private:
  using overlap_fields_tag =
      detail::OverlapFieldsTag<Dim, tmpl::list<OverlapFields...>, OptionsGroup>;

 public:
  using simple_tags =
      tmpl::list<Tags::Overlaps<OverlapFields, Dim, OptionsGroup>...>;
  using const_global_cache_tags =
      tmpl::list<Tags::MaxOverlap<OptionsGroup>,
                 logging::Tags::Verbosity<OptionsGroup>>;
  using inbox_tags = tmpl::list<overlap_fields_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<Dim>& element_id, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    const auto& iteration_id =
        get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    const auto& element = get<domain::Tags::Element<Dim>>(box);

    // Nothing to receive if overlap is empty
    if (UNLIKELY(db::get<Tags::MaxOverlap<OptionsGroup>>(box) == 0 or
                 element.number_of_neighbors() == 0)) {
      return {std::move(box), Parallel::AlgorithmExecution::Continue};
    }

    if (not dg::has_received_from_all_mortars<overlap_fields_tag>(
            iteration_id, element, inboxes)) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s(%zu): Receive overlap fields\n", element_id,
                       Options::name<OptionsGroup>(), iteration_id);
    }

    // Move received overlap data into DataBox
    auto received_overlap_fields =
        std::move(tuples::get<overlap_fields_tag>(inboxes)
                      .extract(iteration_id)
                      .mapped());
    db::mutate<Tags::Overlaps<OverlapFields, Dim, OptionsGroup>...>(
        make_not_null(&box),
        [&received_overlap_fields](const auto... local_overlap_fields) {
          if constexpr (sizeof...(OverlapFields) > 1) {
            for (auto& [overlap_id, overlap_fields] : received_overlap_fields) {
              expand_pack((*local_overlap_fields)[overlap_id] =
                              std::move(get<OverlapFields>(overlap_fields))...);
            }
          } else {
            expand_pack((*local_overlap_fields =
                             std::move(received_overlap_fields))...);
          }
        });

    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};
/// \endcond

}  // namespace Actions
}  // namespace LinearSolver::Schwarz
