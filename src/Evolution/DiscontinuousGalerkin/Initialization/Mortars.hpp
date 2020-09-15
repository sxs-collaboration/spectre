// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::Initialization {
/*!
 * \brief Initialize mortars between elements for exchanging boundary correction
 * terms.
 *
 * If the template parameter `AddFluxBoundaryConditionMortars`
 * is set to `false` then the mortar data for flux boundary conditions are not
 * initialized and other boundary conditions can be applied. In this case, the
 * `Tags::Mortar*` tags have no entries for external boundary directions.
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
 *   - `Tags::MortarData<Dim>`
 *   - `Tags::MortarMesh<Dim>`
 *   - `Tags::MortarSize<Dim>`
 *   - `Tags::MortarNextTemporalId<Dim>`
 * - Removes: nothing
 * - Modifies: nothing
 */
template <size_t Dim, bool AddFluxBoundaryConditionMortars = true>
struct Mortars {
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;

  template <typename MappedType>
  using MortarMap = std::unordered_map<Key, MappedType, boost::hash<Key>>;

 public:
  using initialization_tags = tmpl::list<::domain::Tags::InitialExtents<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (db::tag_is_retrievable_v<::domain::Tags::InitialExtents<Dim>,
                                           db::DataBox<DbTagsList>> and
                  not db::tag_is_retrievable_v<Tags::MortarData<Dim>,
                                               db::DataBox<DbTagsList>> and
                  not db::tag_is_retrievable_v<Tags::MortarMesh<Dim>,
                                               db::DataBox<DbTagsList>> and
                  not db::tag_is_retrievable_v<Tags::MortarSize<Dim>,
                                               db::DataBox<DbTagsList>> and
                  not db::tag_is_retrievable_v<Tags::MortarNextTemporalId<Dim>,
                                               db::DataBox<DbTagsList>>) {
      auto [mortar_data, mortar_meshes, mortar_sizes,
            mortar_next_temporal_ids] =
          apply_impl(db::get<::domain::Tags::InitialExtents<Dim>>(box),
                     db::get<::domain::Tags::Element<Dim>>(box),
                     db::get<::Tags::TimeStepId>(box),
                     db::get<::domain::Tags::Interface<
                         ::domain::Tags::InternalDirections<Dim>,
                         ::domain::Tags::Mesh<Dim - 1>>>(box),
                     db::get<::domain::Tags::Interface<
                         ::domain::Tags::BoundaryDirectionsInterior<Dim>,
                         ::domain::Tags::Mesh<Dim - 1>>>(box));
      return std::make_tuple(
          db::create_from<
              db::RemoveTags<>,
              db::AddSimpleTags<Tags::MortarData<Dim>, Tags::MortarMesh<Dim>,
                                Tags::MortarSize<Dim>,
                                Tags::MortarNextTemporalId<Dim>>>(
              std::move(box), std::move(mortar_data), std::move(mortar_meshes),
              std::move(mortar_sizes), std::move(mortar_next_temporal_ids)));
    } else {
      if (not db::tag_is_retrievable_v<::domain::Tags::InitialExtents<Dim>,
                                       db::DataBox<DbTagsList>>) {
        ERROR(
            "Missing a tag in the DataBox. Did you forget to terminate the "
            "phase after removing options? The missing tag is "
            "'domain::Tags::InitialExtents<Dim>'.");
      }
      ERROR(
          "One of the tags being added already exists in the DataBox. The tags "
          "being added are: Tags::MortarData, Tags::MortarMesh, "
          "Tags::MortarSize, Tags::MortarNextTemporalId");
      return std::forward_as_tuple(std::move(box));
    }
  }

 private:
  static std::tuple<MortarMap<evolution::dg::MortarData<Dim>>,
                    MortarMap<Mesh<Dim - 1>>,
                    MortarMap<std::array<Spectral::MortarSize, Dim - 1>>,
                    MortarMap<TimeStepId>>
  apply_impl(
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const Element<Dim>& element, const TimeStepId& next_temporal_id,
      const std::unordered_map<Direction<Dim>, Mesh<Dim - 1>>& interface_meshes,
      const std::unordered_map<Direction<Dim>, Mesh<Dim - 1>>&
          boundary_meshes) noexcept;
};
}  // namespace evolution::dg::Initialization
