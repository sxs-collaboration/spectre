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
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
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
 * For asynchronous boundary schemes, e.g. local time-stepping, this action also
 * initializes the `BoundaryScheme::receive_temporal_id` on mortars (see
 * `dg::Actions::ReceiveDataForFluxes`). It assumes that the current state is
 * _before_ any mortar communications take place and the first communication
 * will happen at what is currently the `receive_temporal_id`.
 *
 * Uses:
 * - DataBox:
 *   - `Tags::Element<Dim>`
 *   - `Tags::Mesh<Dim>`
 *   - `BoundaryScheme::receive_temporal_id`
 *   - `Tags::Interface<Tags::InternalDirections<Dim>, Tags::Mesh<Dim - 1>>`
 *   - `Tags::Interface<
 *   Tags::BoundaryDirectionsInterior<Dim>, Tags::Mesh<Dim - 1>>`
 *
 * DataBox changes:
 * - Adds:
 *   - `Tags::Mortars<BoundaryScheme::mortar_data_tag, dim>`
 *   - `Tags::Mortars<BoundaryScheme::receive_temporal_id, dim>`
 *   - `Tags::Mortars<Tags::Mesh<dim - 1>, dim>`
 *   - `Tags::Mortars<Tags::MortarSize<dim - 1>, dim>`
 * - Removes: nothing
 * - Modifies: nothing
 */
template <typename BoundaryScheme, bool AddFluxBoundaryConditionMortars = true>
struct InitializeMortars {
 private:
  static constexpr size_t dim = BoundaryScheme::volume_dim;
  using mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, dim>;

 public:
  using initialization_tags = tmpl::list<domain::Tags::InitialExtents<dim>>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<db::tag_is_retrievable_v<domain::Tags::InitialExtents<dim>,
                                              DataBox>> = nullptr>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& element = db::get<domain::Tags::Element<dim>>(box);
    const auto& initial_extents =
        db::get<domain::Tags::InitialExtents<dim>>(box);

    typename mortar_data_tag::type mortar_data{};
    typename ::Tags::Mortars<domain::Tags::Mesh<dim - 1>, dim>::type
        mortar_meshes{};
    typename ::Tags::Mortars<::Tags::MortarSize<dim - 1>, dim>::type
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
      }
    }

    // We don't need to initialize the "next" temporal id on mortars if data is
    // communicated synchronously
    using need_next_temporal_id_on_mortars = std::negation<
        std::is_same<typename BoundaryScheme::temporal_id_tag,
                     typename BoundaryScheme::receive_temporal_id_tag>>;
    auto temporal_id_box = initialize_next_temporal_id_on_mortars(
        std::move(box), need_next_temporal_id_on_mortars{});
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeMortars,
            db::AddSimpleTags<
                mortar_data_tag,
                ::Tags::Mortars<domain::Tags::Mesh<dim - 1>, dim>,
                ::Tags::Mortars<::Tags::MortarSize<dim - 1>, dim>>>(
            std::move(temporal_id_box), std::move(mortar_data),
            std::move(mortar_meshes), std::move(mortar_sizes)));
  }

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not db::tag_is_retrievable_v<
                domain::Tags::InitialExtents<dim>, DataBox>> = nullptr>
  static std::tuple<DataBox&&> apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Dependencies not fulfilled. Did you forget to terminate the phase "
        "after removing options?");
  }

 private:
  template <typename DbTagsList>
  static auto initialize_next_temporal_id_on_mortars(
      db::DataBox<DbTagsList>&& box,
      std::true_type /* perform_initialization */) noexcept {
    using next_temporal_id_tag =
        typename BoundaryScheme::receive_temporal_id_tag;
    using mortars_next_temporal_id_tag =
        ::Tags::Mortars<next_temporal_id_tag, dim>;
    const auto& next_temporal_id = get<next_temporal_id_tag>(box);
    typename mortars_next_temporal_id_tag::type mortar_next_temporal_ids{};
    // Since no communication needs to happen for boundary conditions
    // the temporal id is not advanced on the boundary, so we only need to
    // initialize it on internal boundaries
    for (const auto& direction_and_neighbors :
         db::get<domain::Tags::Element<dim>>(box).neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const auto& neighbors = direction_and_neighbors.second;
      for (const auto& neighbor : neighbors) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        mortar_next_temporal_ids.insert({mortar_id, next_temporal_id});
      }
    }
    return ::Initialization::merge_into_databox<
        InitializeMortars, db::AddSimpleTags<mortars_next_temporal_id_tag>>(
        std::move(box), std::move(mortar_next_temporal_ids));
  }

  template <typename DbTagsList>
  static db::DataBox<DbTagsList> initialize_next_temporal_id_on_mortars(
      db::DataBox<DbTagsList>&& box,
      std::false_type /* perform_initialization */) noexcept {
    return std::move(box);
  }
};
}  // namespace Actions
}  // namespace dg
