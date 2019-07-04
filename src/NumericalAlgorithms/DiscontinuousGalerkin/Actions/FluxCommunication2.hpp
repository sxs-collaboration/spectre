// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines actions SendDataForFluxes and ReceiveDataForFluxes

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Magnitude;
template <typename Tag>
struct Next;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace dg {

/// The inbox tag for flux communication.
template <typename BoundaryScheme>
struct FluxesInboxTag {
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id = db::item_type<typename BoundaryScheme::temporal_id_tag>;
  using type = std::map<
      temporal_id,
      FixedHashMap<maximum_number_of_neighbors(volume_dim),
                   std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
                   std::pair<temporal_id, typename BoundaryScheme::RemoteData>,
                   boost::hash<std::pair<Direction<volume_dim>,
                                         ElementId<volume_dim>>>>>;
};

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Receive boundary data needed for fluxes from neighbors.
///
/// Uses:
/// - DataBox:
///   - BoundaryScheme::temporal_id
///   - Tags::Next<BoundaryScheme::temporal_id>
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::Mortars<Tags::Next<BoundaryScheme::temporal_id>, volume_dim>
///   - Tags::Mortars<BoundaryScheme::mortar_data_tag, volume_dim>
///
/// \see SendDataForFluxes
template <typename BoundaryScheme>
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
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using neighbor_temporal_id_tag =
        Tags::Mortars<Tags::Next<temporal_id_tag>, volume_dim>;
    db::mutate<all_mortar_data_tag, neighbor_temporal_id_tag>(
        make_not_null(&box),
        [&inboxes](const gsl::not_null<db::item_type<all_mortar_data_tag>*>
                       mortar_data,
                   const gsl::not_null<db::item_type<neighbor_temporal_id_tag>*>
                       neighbor_next_temporal_ids,
                   const db::item_type<Tags::Next<temporal_id_tag>>&
                       local_next_temporal_id) noexcept {
          auto& inbox = tuples::get<fluxes_inbox_tag>(inboxes);
          for (auto received_data = inbox.begin();
               received_data != inbox.end() and
               received_data->first < local_next_temporal_id;
               received_data = inbox.erase(received_data)) {
            const auto& receive_temporal_id = received_data->first;
            for (auto& received_mortar_data : received_data->second) {
              const auto mortar_id = received_mortar_data.first;
              ASSERT(neighbor_next_temporal_ids->at(mortar_id) ==
                         receive_temporal_id,
                     "Expected data at "
                         << neighbor_next_temporal_ids->at(mortar_id)
                         << " but received at " << receive_temporal_id);
              neighbor_next_temporal_ids->at(mortar_id) =
                  received_mortar_data.second.first;
              mortar_data->at(mortar_id).remote_insert(
                  receive_temporal_id,
                  std::move(received_mortar_data.second.second));
            }
          }

          // The apparently pointless lambda wrapping this check
          // prevents gcc-7.3.0 from segfaulting.
          ASSERT(([
                   &neighbor_next_temporal_ids, &local_next_temporal_id
                 ]() noexcept {
                   return std::all_of(
                       neighbor_next_temporal_ids->begin(),
                       neighbor_next_temporal_ids->end(),
                       [&local_next_temporal_id](const auto& next) noexcept {
                         return next.first.second ==
                                    ElementId<
                                        volume_dim>::external_boundary_id() or
                                next.second >= local_next_temporal_id;
                       });
                 }()),
                 "apply called before all data received");
          ASSERT(
              inbox.empty() or (inbox.size() == 1 and
                                inbox.begin()->first == local_next_temporal_id),
              "Shouldn't have received data that depended upon the step being "
              "taken: Received data at "
                  << inbox.begin()->first << " while stepping to "
                  << local_next_temporal_id);
        },
        db::get<Tags::Next<temporal_id_tag>>(box));

    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox = tuples::get<fluxes_inbox_tag>(inboxes);
    const auto& local_next_temporal_id =
        db::get<Tags::Next<temporal_id_tag>>(box);
    const auto& mortars_next_temporal_id =
        db::get<Tags::Mortars<Tags::Next<temporal_id_tag>, volume_dim>>(box);
    for (const auto& mortar_id_next_temporal_id : mortars_next_temporal_id) {
      const auto& mortar_id = mortar_id_next_temporal_id.first;
      // If on an external boundary
      if (mortar_id.second == ElementId<volume_dim>::external_boundary_id()) {
        continue;
      }
      auto next_temporal_id = mortar_id_next_temporal_id.second;
      while (next_temporal_id < local_next_temporal_id) {
        const auto temporal_received = inbox.find(next_temporal_id);
        if (temporal_received == inbox.end()) {
          return false;
        }
        const auto mortar_received = temporal_received->second.find(mortar_id);
        if (mortar_received == temporal_received->second.end()) {
          return false;
        }
        next_temporal_id = mortar_received->second.first;
      }
    }
    return true;
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Send local boundary data needed for fluxes to neighbors.
///
/// With:
/// - `Interface<Tag> =
///   Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>`
///
/// Uses:
/// - DataBox:
///   - Tags::Element<volume_dim>
///   - BoundaryScheme::temporal_id_tag
///   - Tags::Next<BoundaryScheme::temporal_id_tag>
///   - Interface<Tags::Mesh<volume_dim - 1>>
///   - Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>
///   - Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>
///   - Interface<BoundaryScheme::packaged_remote_data_tag>
///   - Interface<BoundaryScheme::packaged_local_data_tag>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::Mortars<BoundaryScheme::mortar_data_tag, volume_dim>
///
/// \see ReceiveDataForFluxes
template <typename BoundaryScheme>
struct SendDataForFluxes {
 private:
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using fluxes_inbox_tag = dg::FluxesInboxTag<BoundaryScheme>;

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<temporal_id_tag>(box);
    const auto& next_temporal_id = db::get<Tags::Next<temporal_id_tag>>(box);
    const auto& mortar_meshes =
        db::get<Tags::Mortars<Tags::Mesh<volume_dim - 1>, volume_dim>>(box);
    const auto& mortar_sizes =
        db::get<Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>>(
            box);
    const auto& face_meshes =
        db::get<Tags::Interface<Tags::InternalDirections<volume_dim>,
                                Tags::Mesh<volume_dim - 1>>>(box);
    const auto& packaged_remote_data = db::get<
        Tags::Interface<Tags::InternalDirections<volume_dim>,
                        typename BoundaryScheme::packaged_remote_data_tag>>(
        box);
    const auto& packaged_local_data = db::get<
        Tags::Interface<Tags::InternalDirections<volume_dim>,
                        typename BoundaryScheme::packaged_local_data_tag>>(box);

    // Iterate over neighbors
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_and_neighbors.second;
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

      for (const auto& neighbor : neighbors_in_direction) {
        const auto mortar_id = std::make_pair(direction, neighbor);

        // Project the packaged face data to this mortar
        auto remote_mortar_data =
            packaged_remote_data.at(direction).project_to_mortar(
                face_meshes.at(direction), mortar_meshes.at(mortar_id),
                mortar_sizes.at(mortar_id));
        auto local_mortar_data =
            packaged_local_data.at(direction).project_to_mortar(
                face_meshes.at(direction), mortar_meshes.at(mortar_id),
                mortar_sizes.at(mortar_id));

        // Reorient the variables to the neighbor orientation
        if (not orientation.is_aligned()) {
          remote_mortar_data.orient_on_slice(
              mortar_meshes.at(mortar_id).extents(), dimension, orientation);
        }

        // Send remote mortar data to neighbor
        Parallel::receive_data<fluxes_inbox_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                std::make_pair(next_temporal_id,
                               std::move(remote_mortar_data))));

        // Store local mortar data in DataBox
        using all_mortar_data_tag =
            ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag,
                            volume_dim>;
        db::mutate<all_mortar_data_tag>(
            make_not_null(&box),
            [&mortar_id, &temporal_id, &local_mortar_data ](
                const gsl::not_null<db::item_type<all_mortar_data_tag>*>
                    all_mortar_data) noexcept {
              all_mortar_data->at(mortar_id).local_insert(
                  temporal_id, std::move(local_mortar_data));
            });
      }  // loop over neighbors_in_direction
    }    // loop over element.neighbors()

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
