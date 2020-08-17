// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace dg {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Collect data that is needed to compute numerical fluxes and store it
 * on mortars, projecting it if necessary.
 *
 * Set the `DirectionsTag` template parameter to
 * `domain::Tags::InternalDirections` to collect data on internal element
 * interfaces, so they can be communicated to the element's neighbors in a
 * subsequent call to `dg::Actions::SendDataForFluxes`.
 *
 * Alternatively, set the `DirectionsTag` to
 * `domain::Tags::BoundaryDirectionsInterior` to collect data on external
 * element boundaries for imposing boundary conditions through numerical fluxes.
 * In this case, make sure mortars on external boundaries are initialized e.g.
 * by `dg::Actions::InitializeMortars`. Also, make sure the field data on
 * exterior ("ghost") element boundaries (see
 * `domain::Tags::BoundaryDirectionsExterior`) has been updated to represent the
 * boundary conditions before invoking this action.
 *
 * Design decisions:
 *
 * - We assume that all data that is needed on an element to compute numerical
 * fluxes is also needed on its neighbor. This is reasonable since numerical
 * fluxes typically satisfy conservation criteria. It is possible that this
 * assumptions leads to slightly more data being communicated than is necessary,
 * e.g. in a strong-form scheme with a numerical flux that does not require all
 * normal-dot-fluxes remotely, but requires them locally to take the difference
 * to the normal-dot-numerical-fluxes. This overhead is probably negligible,
 * since the number of projection and communication steps is more relevant than
 * the amount of data being communicated. This assumption allows us to perform
 * only one projection to the mortar in each step, instead of projecting local
 * and remote data separately.
 * - Terminology: "Boundary data" is data collected on an element interface
 * that can be projected to a mortar. "Mortar data" is a collection of projected
 * boundary data from both elements that touch the mortar.
 */
template <typename BoundaryScheme, typename DirectionsTag>
struct CollectDataForFluxes;

/// \cond
template <typename BoundaryScheme>
struct CollectDataForFluxes<BoundaryScheme, domain::Tags::InternalDirections<
                                                BoundaryScheme::volume_dim>> {
 private:
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>;
  using boundary_data_computer =
      typename BoundaryScheme::boundary_data_computer;

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Collect data on element interfaces
    auto boundary_data_on_interfaces =
        interface_apply<domain::Tags::InternalDirections<volume_dim>,
                        boundary_data_computer>(box);

    // Project collected data to all internal mortars and store in DataBox
    const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);
    const auto& face_meshes = get<
        domain::Tags::Interface<domain::Tags::InternalDirections<volume_dim>,
                                domain::Tags::Mesh<volume_dim - 1>>>(box);
    const auto& mortar_meshes =
        get<Tags::Mortars<domain::Tags::Mesh<volume_dim - 1>, volume_dim>>(box);
    const auto& mortar_sizes =
        get<Tags::Mortars<Tags::MortarSize<volume_dim - 1>, volume_dim>>(box);
    const auto& temporal_id = get<temporal_id_tag>(box);
    for (const auto& direction_and_neighbors : element.neighbors()) {
      const auto& direction = direction_and_neighbors.first;
      const auto& face_mesh = face_meshes.at(direction);
      for (const auto& neighbor : direction_and_neighbors.second) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        const auto& mortar_mesh = mortar_meshes.at(mortar_id);
        const auto& mortar_size = mortar_sizes.at(mortar_id);

        // Project the data from the face to the mortar.
        // Where no projection is necessary we `std::move` the data directly to
        // avoid a copy. We can't move the data or modify it in-place when
        // projecting, because in that case the face may touch two mortars so we
        // need to keep the data around.
        auto boundary_data_on_mortar =
            dg::needs_projection(face_mesh, mortar_mesh, mortar_size)
                ? boundary_data_on_interfaces.at(direction).project_to_mortar(
                      face_mesh, mortar_mesh, mortar_size)
                : std::move(boundary_data_on_interfaces.at(direction));

        // Store the boundary data on this side of the mortar
        db::mutate<all_mortar_data_tag>(
            make_not_null(&box),
            [&mortar_id, &temporal_id, &boundary_data_on_mortar ](
                const gsl::not_null<typename all_mortar_data_tag::type*>
                    all_mortar_data) noexcept {
              all_mortar_data->at(mortar_id).local_insert(
                  temporal_id, std::move(boundary_data_on_mortar));
            });
      }
    }
    return {std::move(box)};
  }
};

template <typename BoundaryScheme>
struct CollectDataForFluxes<
    BoundaryScheme,
    domain::Tags::BoundaryDirectionsInterior<BoundaryScheme::volume_dim>> {
 private:
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>;
  using boundary_data_computer =
      typename BoundaryScheme::boundary_data_computer;

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Collect data on external element boundaries
    auto interior_boundary_data =
        interface_apply<domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                        boundary_data_computer>(box);
    auto exterior_boundary_data =
        interface_apply<domain::Tags::BoundaryDirectionsExterior<volume_dim>,
                        boundary_data_computer>(box);

    // Store the boundary data on mortars so the external boundaries are
    // treated exactly the same as internal boundaries between elements. In
    // particular, this is important where the
    // `BoundaryScheme::mortar_data_tag::type` is not just
    // `SimpleMortarData`, but e.g. keeps track of the mortar history for
    // local time stepping.
    const auto& temporal_id = get<temporal_id_tag>(box);
    for (const auto& direction :
         get<domain::Tags::BoundaryDirectionsInterior<volume_dim>>(box)) {
      const MortarId<volume_dim> mortar_id{
          direction, ::ElementId<volume_dim>::external_boundary_id()};
      db::mutate<all_mortar_data_tag>(
          make_not_null(&box),
          [
            &temporal_id, &mortar_id, &direction, &interior_boundary_data, &
            exterior_boundary_data
          ](const gsl::not_null<typename all_mortar_data_tag::type*>
                all_mortar_data) noexcept {
            // We don't need to project the boundary data since mortars and
            // element faces are identical on external boundaries.
            all_mortar_data->at(mortar_id).local_insert(
                temporal_id, std::move(interior_boundary_data.at(direction)));
            all_mortar_data->at(mortar_id).remote_insert(
                temporal_id, std::move(exterior_boundary_data.at(direction)));
          });
    }

    return {std::move(box)};
  }
};
/// \endcond

}  // namespace Actions
}  // namespace dg
