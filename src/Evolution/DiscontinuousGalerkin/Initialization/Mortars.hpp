// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/BoundaryHistory.hpp"
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
namespace detail {
template <size_t Dim>
std::tuple<
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::MortarData<Dim>,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>,
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>, Mesh<Dim - 1>,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>,
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>,
                       std::array<Spectral::MortarSize, Dim - 1>,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>,
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>, TimeStepId,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>,
    DirectionMap<Dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<Dim>>>>>>
mortars_apply_impl(const std::vector<std::array<size_t, Dim>>& initial_extents,
                   Spectral::Quadrature quadrature, const Element<Dim>& element,
                   const TimeStepId& next_temporal_id,
                   const Mesh<Dim>& volume_mesh) noexcept;
}  // namespace detail

/*!
 * \brief Initialize mortars between elements for exchanging boundary correction
 * terms.
 *
 * Uses:
 * - DataBox:
 *   - `Tags::Element<Dim>`
 *   - `Tags::Mesh<Dim>`
 *   - `BoundaryScheme::receive_temporal_id`
 *
 * DataBox changes:
 * - Adds:
 *   - `Tags::MortarData<Dim>`
 *   - `Tags::MortarMesh<Dim>`
 *   - `Tags::MortarSize<Dim>`
 *   - `Tags::MortarNextTemporalId<Dim>`
 *   - `evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>`
 * - Removes: nothing
 * - Modifies: nothing
 */
template <size_t Dim, typename System>
struct Mortars {
  using Key = std::pair<Direction<Dim>, ElementId<Dim>>;

  template <typename MappedType>
  using MortarMap = std::unordered_map<Key, MappedType, boost::hash<Key>>;

 public:
  using initialization_tags = tmpl::list<::domain::Tags::InitialExtents<Dim>,
                                         evolution::dg::Tags::Quadrature>;

  using simple_tags = tmpl::list<
      Tags::MortarData<Dim>, Tags::MortarMesh<Dim>, Tags::MortarSize<Dim>,
      Tags::MortarNextTemporalId<Dim>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
      Tags::MortarDataHistory<
          Dim, typename db::add_tag_prefix<
                   ::Tags::dt, typename System::variables_tag>::type>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (db::tag_is_retrievable_v<::domain::Tags::InitialExtents<Dim>,
                                           db::DataBox<DbTagsList>>) {
      auto [mortar_data, mortar_meshes, mortar_sizes, mortar_next_temporal_ids,
            normal_covector_quantities] =
          detail::mortars_apply_impl(
              db::get<::domain::Tags::InitialExtents<Dim>>(box),
              db::get<evolution::dg::Tags::Quadrature>(box),
              db::get<::domain::Tags::Element<Dim>>(box),
              db::get<::Tags::TimeStepId>(box),
              db::get<::domain::Tags::Mesh<Dim>>(box));
      typename Tags::MortarDataHistory<
          Dim, typename db::add_tag_prefix<
                   ::Tags::dt, typename System::variables_tag>::type>::type
          boundary_data_history{};
      if (Metavariables::local_time_stepping) {
        for (const auto& mortar_id_and_data : mortar_data) {
          // default initialize data
          boundary_data_history[mortar_id_and_data.first];
        }
      }
      ::Initialization::mutate_assign<simple_tags>(
          make_not_null(&box), std::move(mortar_data), std::move(mortar_meshes),
          std::move(mortar_sizes), std::move(mortar_next_temporal_ids),
          std::move(normal_covector_quantities),
          std::move(boundary_data_history));
      return std::make_tuple(std::move(box));
    } else {
      ERROR(
          "Missing a tag in the DataBox. Did you forget to terminate the "
          "phase after removing options? The missing tag is "
          "'domain::Tags::InitialExtents<Dim>'.");
      return std::forward_as_tuple(std::move(box));
    }
  }
};
}  // namespace evolution::dg::Initialization
