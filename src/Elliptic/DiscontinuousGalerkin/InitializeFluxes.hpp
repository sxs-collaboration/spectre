// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct ConstGlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Initialize DataBox tags related to discontinuous Galerkin fluxes
 *
 * With:
 * - `normal_dot_flux_tag` = `db::add_tag_prefix<Tags::NormalDotFlux,
 * variables_tag>`
 * - `interface<Tag>` =
 * `Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>`
 * - `boundary<Tag>` =
 * `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>`
 *
 * Uses:
 * - Metavariables:
 *   - Items required by `flux_comm_types`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 * - DataBox:
 *   - `Tags::InternalDirections<volume_dim>`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `interface<Tags::Mesh<volume_dim - 1>>`
 *   - `boundary<Tags::Mesh<volume_dim - 1>>`
 *
 * DataBox:
 * - Adds:
 *   - `interface<normal_dot_flux_tag>`
 *   - `boundary<normal_dot_flux_tag>`
 */
template <typename Metavariables>
struct InitializeFluxes {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;
  using normal_dot_flux_tag =
      db::add_tag_prefix<::Tags::NormalDotFlux,
                         typename Metavariables::system::variables_tag>;

  template <typename Tag>
  using interface_tag =
      Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>;
  template <typename Tag>
  using boundary_tag =
      Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& internal_directions =
        db::get<Tags::InternalDirections<volume_dim>>(box);
    const auto& boundary_directions =
        db::get<Tags::BoundaryDirectionsInterior<volume_dim>>(box);

    db::item_type<interface_tag<normal_dot_flux_tag>>
        interface_normal_dot_fluxes{};
    for (const auto& direction : internal_directions) {
      const size_t interface_num_points =
          db::get<interface_tag<Tags::Mesh<volume_dim - 1>>>(box)
              .at(direction)
              .number_of_grid_points();
      interface_normal_dot_fluxes[direction].initialize(interface_num_points,
                                                        0.);
    }
    db::item_type<boundary_tag<normal_dot_flux_tag>>
        boundary_normal_dot_fluxes{};
    for (const auto& direction : boundary_directions) {
      const size_t interface_num_points =
          db::get<boundary_tag<Tags::Mesh<volume_dim - 1>>>(box)
              .at(direction)
              .number_of_grid_points();
      boundary_normal_dot_fluxes[direction].initialize(interface_num_points,
                                                       0.);
    }

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeFluxes,
            db::AddSimpleTags<interface_tag<normal_dot_flux_tag>,
                              boundary_tag<normal_dot_flux_tag>>>(
            std::move(box), std::move(interface_normal_dot_fluxes),
            std::move(boundary_normal_dot_fluxes)));
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
