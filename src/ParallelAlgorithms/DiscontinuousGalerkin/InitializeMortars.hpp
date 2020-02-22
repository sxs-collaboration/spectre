// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/CreateInitialMesh.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace dg {
namespace Actions {
/*!
 * \ingroup InitializationGroup
 * \brief Initialize mortars between elements for exchanging fluxes.
 *
 * If the template parameter `AddFluxBoundaryConditionMortars`
 * is set to `false` then the mortar data for flux boundary conditions are not
 * initialized and other boundary conditions can be applied. In this case, the
 * `Tags::Mortars` items have no entries for external boundary directions.
 *
 * \note This action assumes that the current state is _before_ any mortar
 * communications take place and the first communication will happen at the
 * `Tags::Next<temporal_id>`.
 *
 * Uses:
 * - Metavariables:
 *   - `temporal_id`
 *   - `local_time_stepping`
 * - System:
 *   - `volume_dim`
 * - DataBox:
 *   - `Tags::Element<Dim>`
 *   - `Tags::Mesh<Dim>`
 *   - `Tags::Next<temporal_id>`
 *   - `Tags::Interface<Tags::InternalDirections<Dim>, Tags::Mesh<Dim - 1>>`
 *   - `Tags::Interface<
 *   Tags::BoundaryDirectionsInterior<Dim>, Tags::Mesh<Dim - 1>>`
 *
 * DataBox changes:
 * - Adds:
 *   - Tags::VariablesBoundaryData
 *   - Tags::Mortars<Tags::Next<temporal_id_tag>, dim>
 *   - Tags::Mortars<Tags::Mesh<dim - 1>, dim>
 *   - Tags::Mortars<Tags::MortarSize<dim - 1>, dim>
 * - Removes: nothing
 * - Modifies: nothing
 */
template <typename Metavariables, bool AddFluxBoundaryConditionMortars = true>
struct InitializeMortars {
 private:
  static constexpr size_t dim = Metavariables::system::volume_dim;
  using temporal_id_tag = typename Metavariables::temporal_id;
  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;
  using mortar_data_tag = tmpl::conditional_t<
      Metavariables::local_time_stepping,
      typename flux_comm_types::local_time_stepping_mortar_data_tag,
      typename flux_comm_types::simple_mortar_data_tag>;

 public:
  using initialization_tags = tmpl::list<domain::Tags::InitialExtents<dim>>;

  template <typename DataBox, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<db::tag_is_retrievable_v<domain::Tags::InitialExtents<dim>,
                                              DataBox>> = nullptr>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& element = db::get<domain::Tags::Element<dim>>(box);
    const auto& next_temporal_id = get<::Tags::Next<temporal_id_tag>>(box);
    const auto& initial_extents =
        db::get<domain::Tags::InitialExtents<dim>>(box);

    db::item_type<mortar_data_tag> mortar_data{};
    db::item_type<::Tags::Mortars<::Tags::Next<temporal_id_tag>, dim>>
        mortar_next_temporal_ids{};
    db::item_type<::Tags::Mortars<domain::Tags::Mesh<dim - 1>, dim>>
        mortar_meshes{};
    db::item_type<::Tags::Mortars<::Tags::MortarSize<dim - 1>, dim>>
        mortar_sizes{};
    const auto& interface_meshes =
        db::get<domain::Tags::Interface<domain::Tags::InternalDirections<dim>,
                                        domain::Tags::Mesh<dim - 1>>>(box);
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const auto& neighbors = direction_and_neighbors.second;
      for (const auto& neighbor : neighbors) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        mortar_data[mortar_id];  // Default initialize data
        mortar_next_temporal_ids.insert({mortar_id, next_temporal_id});
        mortar_meshes.emplace(
            mortar_id,
            dg::mortar_mesh(
                interface_meshes.at(direction),
                domain::Initialization::create_initial_mesh(
                    initial_extents, neighbor, neighbors.orientation())
                    .slice_away(direction.dimension())));
        mortar_sizes.emplace(
            mortar_id,
            dg::mortar_size(element.id(), neighbor, direction.dimension(),
                            neighbors.orientation()));
      }
    }

    if (AddFluxBoundaryConditionMortars) {
      const auto& boundary_meshes = db::get<
          domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<dim>,
                                  domain::Tags::Mesh<dim - 1>>>(box);
      for (const auto& direction : element.external_boundaries()) {
        const auto mortar_id =
            std::make_pair(direction, ElementId<dim>::external_boundary_id());
        mortar_data[mortar_id];  // Default initialize data
        mortar_meshes.emplace(mortar_id, boundary_meshes.at(direction));
        mortar_sizes.emplace(mortar_id,
                             make_array<dim - 1>(Spectral::MortarSize::Full));
        // Since no communication needs to happen for boundary conditions
        // the temporal id is not advanced on the boundary, so we don't need to
        // initialize it.
      }
    }

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeMortars,
            db::AddSimpleTags<
                mortar_data_tag,
                ::Tags::Mortars<::Tags::Next<temporal_id_tag>, dim>,
                ::Tags::Mortars<domain::Tags::Mesh<dim - 1>, dim>,
                ::Tags::Mortars<::Tags::MortarSize<dim - 1>, dim>>>(
            std::move(box), std::move(mortar_data),
            std::move(mortar_next_temporal_ids), std::move(mortar_meshes),
            std::move(mortar_sizes)));
  }

  template <typename DataBox, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<not db::tag_is_retrievable_v<
                domain::Tags::InitialExtents<dim>, DataBox>> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Dependencies not fulfilled. Did you forget to terminate the phase "
        "after removing options?");
  }
};
}  // namespace Actions
}  // namespace dg
