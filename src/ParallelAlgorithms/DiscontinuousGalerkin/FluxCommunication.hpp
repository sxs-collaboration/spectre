// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/HasReceivedFromAllMortars.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace dg {

/// The inbox tag for flux communication
template <typename BoundaryScheme>
struct FluxesInboxTag
    : public Parallel::InboxInserters::Map<FluxesInboxTag<BoundaryScheme>> {
 private:
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;

 public:
  using temporal_id = typename BoundaryScheme::temporal_id_tag::type;
  using type = std::map<
      temporal_id,
      FixedHashMap<
          maximum_number_of_neighbors(volume_dim), MortarId<volume_dim>,
          std::pair<temporal_id, typename BoundaryScheme::BoundaryData>,
          boost::hash<MortarId<volume_dim>>>>;
};

namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Send local boundary data needed for fluxes to neighbors.
 *
 * Sends the data in `Tags::Mortars<BoundaryScheme::mortar_data_tag,
 * volume_dim>` to neighbors. Note that this action does not collect the data
 * from element interfaces or compute anything, it only sends the available data
 * to neighbors. Make sure to collect the data that should be sent before
 * invoking this action and store them in
 * `Tags::Mortars<BoundaryScheme::mortar_data_tag, volume_dim>`, including
 * projecting them to the mortar if necessary. See
 * `dg::Actions::CollectDataForFluxes` for an action that uses the interface
 * tag mechanism to do so.
 *
 * Uses:
 * - DataBox:
 *   - `Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>`
 *   - `Tags::Element<volume_dim>`
 *   - `BoundaryScheme::temporal_id_tag`
 *   - `BoundaryScheme::receive_temporal_id_tag`
 *   - `Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>`
 *
 * \see `dg::Actions::ReceiveDataForFluxes`
 */
template <typename BoundaryScheme>
struct SendDataForFluxes {
 private:
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using receive_temporal_id_tag =
      typename BoundaryScheme::receive_temporal_id_tag;
  using fluxes_inbox_tag = dg::FluxesInboxTag<BoundaryScheme>;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>;

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const auto& all_mortar_data = get<all_mortar_data_tag>(box);
    const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<temporal_id_tag>(box);
    const auto& receive_temporal_id = db::get<receive_temporal_id_tag>(box);
    const auto& mortar_meshes = db::get<
        ::Tags::Mortars<domain::Tags::Mesh<volume_dim - 1>, volume_dim>>(box);

    // Iterate over neighbors
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_and_neighbors.second;
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

      for (const auto& neighbor : neighbors_in_direction) {
        const MortarId<volume_dim> mortar_id{direction, neighbor};

        // Make a copy of the local boundary data on the mortar to send to the
        // neighbor
        ASSERT(all_mortar_data.find(mortar_id) != all_mortar_data.end(),
               "Mortar data on mortar "
                   << mortar_id
                   << " not available for sending. Did you forget to collect "
                      "the data on mortars?");
        auto remote_boundary_data_on_mortar =
            all_mortar_data.at(mortar_id).local_data(temporal_id);

        // Reorient the data to the neighbor orientation if necessary
        if (not orientation.is_aligned()) {
          remote_boundary_data_on_mortar.orient_on_slice(
              mortar_meshes.at(mortar_id).extents(), dimension, orientation);
        }

        // Send remote data to neighbor
        Parallel::receive_data<fluxes_inbox_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                MortarId<volume_dim>{direction_from_neighbor, element.id()},
                std::make_pair(receive_temporal_id,
                               std::move(remote_boundary_data_on_mortar))));
      }
    }
    return {std::move(box)};
  }
};

/*!
 * \ingroup ActionsGroup
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Receive boundary data needed for fluxes from neighbors.
 *
 * Waits until remote boundary data from all internal mortars has arrived, then
 * contributes it to the data in `Tags::Mortars<BoundaryScheme::mortar_data_tag,
 * volume_dim>` that already holds the local boundary data.
 *
 * Uses:
 * - DataBox:
 *   - `Tags::Element<volume_dim>`
 *   - `BoundaryScheme::temporal_id`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::Mortars<BoundaryScheme::mortar_data_tag, volume_dim>`
 *
 * \see `dg::Actions::SendDataForFluxes`
 */
template <typename BoundaryScheme, typename = std::nullptr_t>
struct ReceiveDataForFluxes {
 private:
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using fluxes_inbox_tag = dg::FluxesInboxTag<BoundaryScheme>;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>;

 public:
  using inbox_tags = tmpl::list<fluxes_inbox_tag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&, Parallel::AlgorithmExecution> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (not has_received_from_all_mortars<fluxes_inbox_tag>(
            db::get<temporal_id_tag>(box),
            get<domain::Tags::Element<volume_dim>>(box), inboxes)) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    if (UNLIKELY(
            get<domain::Tags::Element<volume_dim>>(box).number_of_neighbors() ==
            0)) {
      return {std::move(box), Parallel::AlgorithmExecution::Continue};
    }

    auto& inbox = tuples::get<fluxes_inbox_tag>(inboxes);
    const auto& temporal_id = get<temporal_id_tag>(box);
    const auto temporal_received = inbox.find(temporal_id);
    db::mutate<all_mortar_data_tag>(
        make_not_null(&box), [&temporal_received](
                                 const gsl::not_null<
                                     typename all_mortar_data_tag::type*>
                                     mortar_data) noexcept {
          for (auto& received_mortar_data : temporal_received->second) {
            const auto& mortar_id = received_mortar_data.first;
            mortar_data->at(mortar_id).remote_insert(
                temporal_received->first,
                std::move(received_mortar_data.second.second));
          }
        });
    inbox.erase(temporal_received);
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

}  // namespace Actions
}  // namespace dg
