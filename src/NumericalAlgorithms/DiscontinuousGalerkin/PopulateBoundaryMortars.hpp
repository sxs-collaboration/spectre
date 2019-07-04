// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "Domain/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {

/*!
 * \brief Populates the mortar data on external boundaries with the packaged
 * data from the interior and the exterior (ghost) faces.
 */
template <typename BoundaryScheme>
struct PopulateBoundaryMortars {
  static constexpr size_t volume_dim = BoundaryScheme::volume_dim;
  using directions_tag = ::Tags::BoundaryDirectionsInterior<volume_dim>;
  using packaged_local_data_tag =
      ::Tags::Interface<::Tags::BoundaryDirectionsInterior<volume_dim>,
                        typename BoundaryScheme::packaged_local_data_tag>;
  using packaged_remote_data_tag =
      ::Tags::Interface<::Tags::BoundaryDirectionsExterior<volume_dim>,
                        typename BoundaryScheme::packaged_remote_data_tag>;
  using temporal_id_tag = typename BoundaryScheme::temporal_id_tag;
  using all_mortar_data_tag =
      ::Tags::Mortars<typename BoundaryScheme::mortar_data_tag, volume_dim>;

  using argument_tags = tmpl::list<directions_tag, packaged_local_data_tag,
                                   packaged_remote_data_tag, temporal_id_tag>;
  using return_tags = tmpl::list<all_mortar_data_tag>;
  static void apply(
      const gsl::not_null<db::item_type<all_mortar_data_tag>*> all_mortar_data,
      const db::item_type<directions_tag>& directions,
      const db::item_type<packaged_local_data_tag>& packaged_local_data,
      const db::item_type<packaged_remote_data_tag>& packaged_remote_data,
      const db::item_type<temporal_id_tag>& temporal_id) noexcept {
    for (const auto& direction : directions) {
      const auto mortar_id = std::make_pair(
          direction, ::ElementId<volume_dim>::external_boundary_id());
      all_mortar_data->at(mortar_id).local_insert(
          temporal_id, packaged_local_data.at(direction));
      all_mortar_data->at(mortar_id).remote_insert(
          temporal_id, packaged_remote_data.at(direction));
    }
  }
};

}  // namespace dg
