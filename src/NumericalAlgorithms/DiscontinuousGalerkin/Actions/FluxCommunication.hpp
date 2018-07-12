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
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
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
namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Receive boundary data needed for fluxes from neighbors.
///
/// Uses:
/// - DataBox:
///   - Metavariables::temporal_id
///   - Tags::Next<Metavariables::temporal_id>
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::Mortars<Tags::Next<Metavariables::temporal_id>, volume_dim>
///   - Tags::VariablesBoundaryData
///
/// \see SendDataForFluxes
template <typename Metavariables>
struct ReceiveDataForFluxes {
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::normal_dot_numerical_flux>;

 private:
  using flux_comm_types = FluxCommunicationTypes<Metavariables>;

 public:
  using inbox_tags = tmpl::list<typename flux_comm_types::FluxesTag>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    using temporal_id_tag = typename Metavariables::temporal_id;
    using neighbor_temporal_id_tag =
        Tags::Mortars<Tags::Next<temporal_id_tag>, volume_dim>;
    db::mutate<Tags::VariablesBoundaryData, neighbor_temporal_id_tag>(
        make_not_null(&box),
        [&inboxes](const gsl::not_null<
                       db::item_type<Tags::VariablesBoundaryData, DbTags>*>
                       mortar_data,
                   const gsl::not_null<db::item_type<neighbor_temporal_id_tag>*>
                       neighbor_next_temporal_ids,
                   const db::item_type<Tags::Next<temporal_id_tag>>&
                       local_next_temporal_id) noexcept {
          auto& inbox =
              tuples::get<typename flux_comm_types::FluxesTag>(inboxes);
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
                         return next.second >= local_next_temporal_id;
                       });
                 }()),
                 "apply called before all data received");
          ASSERT(
              inbox.empty() or (inbox.size() == 1 and
                                inbox.begin()->first == local_next_temporal_id),
              "Shouldn't have received data that depended upon the step being "
              "taken: Received data at " << inbox.begin()->first
              << " while stepping to " << local_next_temporal_id);
        },
        db::get<Tags::Next<temporal_id_tag>>(box));

    return std::forward_as_tuple(std::move(box));
  }

  template <typename DbTags, typename... InboxTags, typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    using temporal_id = typename Metavariables::temporal_id;

    const auto& inbox =
        tuples::get<typename flux_comm_types::FluxesTag>(inboxes);

    const auto& local_next_temporal_id = db::get<Tags::Next<temporal_id>>(box);
    const auto& mortars_next_temporal_id =
        db::get<Tags::Mortars<Tags::Next<temporal_id>, volume_dim>>(box);
    for (const auto& mortar_id_next_temporal_id : mortars_next_temporal_id) {
      const auto& mortar_id = mortar_id_next_temporal_id.first;
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
/// - ConstGlobalCache: Metavariables::normal_dot_numerical_flux
/// - DataBox:
///   - Tags::Element<volume_dim>
///   - Interface<Tags listed in
///               Metavariables::normal_dot_numerical_flux::type::argument_tags>
///   - Interface<Tags::Mesh<volume_dim - 1>>
///   - Interface<Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   - Metavariables::temporal_id
///   - Tags::Next<Metavariables::temporal_id>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::VariablesBoundaryData
///
/// \see ReceiveDataForFluxes
template <typename Metavariables>
struct SendDataForFluxes {
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::normal_dot_numerical_flux>;

  template <typename DbTags, typename... InboxTags,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;

    using flux_comm_types = FluxCommunicationTypes<Metavariables>;

    using interface_normal_dot_fluxes_tag =
        Tags::Interface<Tags::InternalDirections<volume_dim>,
                        typename flux_comm_types::normal_dot_fluxes_tag>;

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<typename Metavariables::temporal_id>(box);
    const auto& next_temporal_id =
        db::get<Tags::Next<typename Metavariables::temporal_id>>(box);

    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_neighbors.second;
      ASSERT(neighbors_in_direction.size() == 1,
             "h-adaptivity is not supported yet.\nDirection: "
                 << direction << "\nDimension: " << dimension
                 << "\nNeighbors:\n"
                 << neighbors_in_direction);
      const auto& orientation = neighbors_in_direction.orientation();
      const auto& boundary_mesh =
          db::get<Tags::Interface<Tags::InternalDirections<volume_dim>,
                                  Tags::Mesh<volume_dim - 1>>>(box)
              .at(direction);

      // Everything below here needs to be fixed for
      // hp-adaptivity to handle projections correctly

      // We compute the parts of the numerical flux that only depend on data
      // from this side of the mortar now, then package it into a Variables.
      // We store one copy of the Variables and send another, since we need
      // the data on both sides of the mortar.
      using package_arguments = typename Metavariables::
          normal_dot_numerical_flux::type::argument_tags;
      const auto packaged_data = db::apply<tmpl::transform<
          package_arguments,
          tmpl::bind<Tags::Interface, Tags::InternalDirections<volume_dim>,
                     tmpl::_1>>>(
          [&boundary_mesh, &direction,
           &normal_dot_numerical_flux_computer ](const auto&... args) noexcept {
            typename flux_comm_types::PackagedData ret(
                boundary_mesh.number_of_grid_points(), 0.0);
            normal_dot_numerical_flux_computer.package_data(
                make_not_null(&ret), args.at(direction)...);
            return ret;
          },
          box);

      typename flux_comm_types::LocalData local_data(
          boundary_mesh.number_of_grid_points());
      local_data.assign_subset(
          db::get<interface_normal_dot_fluxes_tag>(box).at(direction));
      local_data.assign_subset(packaged_data);
      get<typename flux_comm_types::MagnitudeOfFaceNormal>(local_data) =
          db::get<Tags::Interface<
              Tags::InternalDirections<volume_dim>,
              Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>>(box)
              .at(direction);

      const auto direction_from_neighbor = orientation(direction.opposite());

      // orient_variables_on_slice only needs to be done in the case where
      // the data is oriented differently. This needs to improved later.
      // Note: avoiding the same-orientation-on-both-sides copy is possible
      // even with AMR since the quantities are already on the mortar at this
      // point
      const auto neighbor_packaged_data = orient_variables_on_slice(
          packaged_data, boundary_mesh.extents(), dimension, orientation);

      for (const auto& neighbor : neighbors_in_direction) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        Parallel::receive_data<typename flux_comm_types::FluxesTag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                std::make_pair(next_temporal_id, neighbor_packaged_data)));

        db::mutate<Tags::VariablesBoundaryData>(
            make_not_null(&box),
            [&mortar_id, &temporal_id, &local_data](
                const gsl::not_null<
                    db::item_type<Tags::VariablesBoundaryData, DbTags>*>
                    mortar_data) noexcept {
              mortar_data->at(mortar_id).local_insert(temporal_id, local_data);
            });
      }  // loop over neighbors_in_direction
    }    // loop over element.neighbors()

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
