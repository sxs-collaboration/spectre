// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Conservative/Tags.hpp"
#include "Evolution/Initialization/Helpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Initialization {

/// \brief Initialize items related to the discontinuous Galerkin method
///
/// Uses:
/// - DataBox:
///   * `Tags::Element<Dim>`
///   * `Tags::Mesh<Dim>`
///   * `Tags::Next<temporal_id_tag>`
///   * `Tags::Interface<Tags::InternalDirections<Dim>, Tags::Mesh<Dim - 1>>`
///
/// DataBox changes:
/// - Adds:
///   * Tags::Interface<Tags::InternalDirections<Dim>,
///                     typename flux_comm_types::normal_dot_fluxes_tag>
///   * mortar_data_tag
///   * Tags::Mortars<Tags::Next<temporal_id_tag>, dim>
///   * Tags::Mortars<Tags::Mesh<dim - 1>, dim>
///   * Tags::Mortars<Tags::MortarSize<dim - 1>, dim>
/// - Removes: nothing
/// - Modifies: nothing
template <typename Metavariables>
struct DiscontinuousGalerkin {
  static constexpr size_t dim = Metavariables::system::volume_dim;
  using temporal_id_tag = typename Metavariables::temporal_id;
  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;
  using mortar_data_tag = tmpl::conditional_t<
      Metavariables::local_time_stepping,
      typename flux_comm_types::local_time_stepping_mortar_data_tag,
      typename flux_comm_types::simple_mortar_data_tag>;

  template <typename Tag>
  using interface_tag = Tags::Interface<Tags::InternalDirections<dim>, Tag>;

  template <typename TagsList>
  static auto add_mortar_data(
      db::DataBox<TagsList>&& box,
      const std::vector<std::array<size_t, dim>>& initial_extents) noexcept {
    const auto& element = db::get<Tags::Element<dim>>(box);
    const auto& mesh = db::get<Tags::Mesh<dim>>(box);

    typename mortar_data_tag::type mortar_data{};
    typename Tags::Mortars<Tags::Next<temporal_id_tag>, dim>::type
        mortar_next_temporal_ids{};
    typename Tags::Mortars<Tags::Mesh<dim - 1>, dim>::type mortar_meshes{};
    typename Tags::Mortars<Tags::MortarSize<dim - 1>, dim>::type mortar_sizes{};
    const auto& temporal_id = get<Tags::Next<temporal_id_tag>>(box);
    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const auto& neighbors = direction_neighbors.second;
      for (const auto& neighbor : neighbors) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        mortar_data[mortar_id];  // Default initialize data
        mortar_next_temporal_ids.insert({mortar_id, temporal_id});
        mortar_meshes.emplace(
            mortar_id, dg::mortar_mesh(mesh.slice_away(direction.dimension()),
                                       element_mesh(initial_extents, neighbor,
                                                    neighbors.orientation())
                                           .slice_away(direction.dimension())));
        mortar_sizes.emplace(
            mortar_id,
            dg::mortar_size(element.id(), neighbor, direction.dimension(),
                            neighbors.orientation()));
      }
    }

    return db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<mortar_data_tag,
                          Tags::Mortars<Tags::Next<temporal_id_tag>, dim>,
                          Tags::Mortars<Tags::Mesh<dim - 1>, dim>,
                          Tags::Mortars<Tags::MortarSize<dim - 1>, dim>>>(
        std::move(box), std::move(mortar_data),
        std::move(mortar_next_temporal_ids), std::move(mortar_meshes),
        std::move(mortar_sizes));
  }

  template <typename LocalSystem,
            bool IsConservative = LocalSystem::is_conservative>
  struct Impl {
    using simple_tags = db::AddSimpleTags<
        mortar_data_tag, Tags::Mortars<Tags::Next<temporal_id_tag>, dim>,
        Tags::Mortars<Tags::Mesh<dim - 1>, dim>,
        Tags::Mortars<Tags::MortarSize<dim - 1>, dim>,
        interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>>;

    using compute_tags = db::AddComputeTags<>;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const std::vector<std::array<size_t, dim>>& initial_extents) noexcept {
      auto box2 = add_mortar_data(std::move(box), initial_extents);

      const auto& internal_directions =
          db::get<Tags::InternalDirections<dim>>(box2);

      typename interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>::
          type normal_dot_fluxes{};
      for (const auto& direction : internal_directions) {
        const auto& interface_num_points =
            db::get<interface_tag<Tags::Mesh<dim - 1>>>(box2)
                .at(direction)
                .number_of_grid_points();
        normal_dot_fluxes[direction].initialize(interface_num_points, 0.);
      }

      return db::create_from<
          db::RemoveTags<>,
          db::AddSimpleTags<
              interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>>>(
          std::move(box2), std::move(normal_dot_fluxes));
    }
  };

  template <typename LocalSystem>
  struct Impl<LocalSystem, true> {
    using simple_tags =
        db::AddSimpleTags<mortar_data_tag,
                          Tags::Mortars<Tags::Next<temporal_id_tag>, dim>,
                          Tags::Mortars<Tags::Mesh<dim - 1>, dim>,
                          Tags::Mortars<Tags::MortarSize<dim - 1>, dim>>;

    template <typename Tag>
    using interface_compute_tag =
        Tags::InterfaceComputeItem<Tags::InternalDirections<dim>, Tag>;

    template <typename Tag>
    using boundary_interior_compute_tag =
        Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<dim>, Tag>;

    template <typename Tag>
    using boundary_exterior_compute_tag =
        Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<dim>, Tag>;

    using char_speed_tag = typename LocalSystem::char_speeds_tag;

    using compute_tags = db::AddComputeTags<
        Tags::Slice<
            Tags::InternalDirections<dim>,
            db::add_tag_prefix<Tags::Flux, typename LocalSystem::variables_tag,
                               tmpl::size_t<dim>, Frame::Inertial>>,
        interface_compute_tag<Tags::ComputeNormalDotFlux<
            typename LocalSystem::variables_tag, dim, Frame::Inertial>>,
        interface_compute_tag<char_speed_tag>,
        Tags::Slice<
            Tags::BoundaryDirectionsInterior<dim>,
            db::add_tag_prefix<Tags::Flux, typename LocalSystem::variables_tag,
                               tmpl::size_t<dim>, Frame::Inertial>>,
        boundary_interior_compute_tag<Tags::ComputeNormalDotFlux<
            typename LocalSystem::variables_tag, dim, Frame::Inertial>>,
        Tags::Slice<Tags::BoundaryDirectionsExterior<dim>,
                    db::add_tag_prefix<Tags::Flux,
                                       typename LocalSystem::variables_tag,
                                       tmpl::size_t<dim>, Frame::Inertial>>,
        boundary_exterior_compute_tag<Tags::ComputeNormalDotFlux<
            typename LocalSystem::variables_tag, dim, Frame::Inertial>>>;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const std::vector<std::array<size_t, dim>>& initial_extents) noexcept {
      return db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                             compute_tags>(
          add_mortar_data(std::move(box), initial_extents));
    }
  };

  using impl = Impl<typename Metavariables::system>;
  using simple_tags = typename impl::simple_tags;
  using compute_tags = typename impl::compute_tags;

  template <typename TagsList>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const std::vector<std::array<size_t, dim>>& initial_extents) noexcept {
    return impl::initialize(std::move(box), initial_extents);
  }
};

}  // namespace Initialization
