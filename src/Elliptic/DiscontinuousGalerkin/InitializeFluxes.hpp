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
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic {
namespace Initialization {

/*!
 * \brief Initializes DataBox tags related to discontinuous Galerkin fluxes
 *
 * With:
 * - `flux_comm_types` = `dg::FluxCommunicationTypes<Metavariables>`
 * - `mortar_data_tag` = `flux_comm_types::simple_mortar_data_tag`
 * - `interface<Tag>` =
 * `Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>`
 * - `boundary<Tag>` =
 * `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>`
 * - `mortar<Tag>` = `Tags::Mortars<Tag, volume_dim>`
 *
 * Uses:
 * - Metavariables:
 *   - `temporal_id`
 *   - Items required by `flux_comm_types`
 * - System:
 *   - `volume_dim`
 * - DataBox:
 *   - `Tags::Element<volume_dim>`
 *   - `Tags::Mesh<volume_dim>`
 *   - `temporal_id`
 *   - `Tags::InternalDirections<volume_dim>`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `interface<Tags::Mesh<volume_dim - 1>>`
 *   - `boundary<Tags::Mesh<volume_dim - 1>>`
 *
 * DataBox:
 * - Adds:
 *   - `mortar_data_tag`
 *   - `mortar<Tags::Next<temporal_id>>`
 *   - `mortar<Tags::Mesh<volume_dim - 1>>`
 *   - `mortar<Tags::MortarSize<volume_dim - 1>>`
 *   - `interface<flux_comm_types::normal_dot_fluxes_tag>`
 *   - `boundary<flux_comm_types::normal_dot_fluxes_tag>`
 */
template <typename Metavariables>
struct DiscontinuousGalerkin {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;
  using flux_comm_types = ::dg::FluxCommunicationTypes<Metavariables>;

  template <typename Tag>
  using interface_tag =
      Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>;
  template <typename Tag>
  using boundary_tag =
      Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>;

  using simple_tags = db::AddSimpleTags<
      interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>,
      boundary_tag<typename flux_comm_types::normal_dot_fluxes_tag>>;

  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    const auto& internal_directions =
        db::get<Tags::InternalDirections<volume_dim>>(box);
    const auto& boundary_directions =
        db::get<Tags::BoundaryDirectionsInterior<volume_dim>>(box);

    db::item_type<
        interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>>
        normal_dot_fluxes{};
    for (const auto& direction : internal_directions) {
      const auto& interface_num_points =
          db::get<interface_tag<Tags::Mesh<volume_dim - 1>>>(box)
              .at(direction)
              .number_of_grid_points();
      normal_dot_fluxes[direction].initialize(interface_num_points, 0.);
    }
    db::item_type<boundary_tag<typename flux_comm_types::normal_dot_fluxes_tag>>
        boundary_normal_dot_fluxes{};
    for (const auto& direction : boundary_directions) {
      const auto& interface_num_points =
          db::get<boundary_tag<Tags::Mesh<volume_dim - 1>>>(box)
              .at(direction)
              .number_of_grid_points();
      boundary_normal_dot_fluxes[direction].initialize(interface_num_points,
                                                       0.);
    }

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(normal_dot_fluxes),
        std::move(boundary_normal_dot_fluxes));
  }
};
}  // namespace Initialization
}  // namespace elliptic
