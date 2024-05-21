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
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Spectral {
enum class ChildSize;
using MortarSize = ChildSize;
enum class Quadrature : uint8_t;
}  // namespace Spectral
namespace Tags {
struct TimeStepId;
}  // namespace Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::Initialization {
namespace detail {
template <size_t Dim>
std::tuple<
    std::unordered_map<DirectionalId<Dim>, evolution::dg::MortarData<Dim>,
                       boost::hash<DirectionalId<Dim>>>,
    std::unordered_map<DirectionalId<Dim>, Mesh<Dim - 1>,
                       boost::hash<DirectionalId<Dim>>>,
    std::unordered_map<DirectionalId<Dim>,
                       std::array<Spectral::MortarSize, Dim - 1>,
                       boost::hash<DirectionalId<Dim>>>,
    std::unordered_map<DirectionalId<Dim>, TimeStepId,
                       boost::hash<DirectionalId<Dim>>>,
    DirectionMap<Dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<Dim>>>>>>
mortars_apply_impl(const std::vector<std::array<size_t, Dim>>& initial_extents,
                   Spectral::Quadrature quadrature, const Element<Dim>& element,
                   const TimeStepId& next_temporal_id,
                   const Mesh<Dim>& volume_mesh);
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
  using Key = DirectionalId<Dim>;

  template <typename MappedType>
  using MortarMap = std::unordered_map<Key, MappedType, boost::hash<Key>>;

 public:
  using simple_tags_from_options =
      tmpl::list<::domain::Tags::InitialExtents<Dim>,
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
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto [mortar_data, mortar_meshes, mortar_sizes, mortar_next_temporal_ids,
          normal_covector_quantities] =
        detail::mortars_apply_impl(
            db::get<::domain::Tags::InitialExtents<Dim>>(box),
            db::get<evolution::dg::Tags::Quadrature>(box),
            db::get<::domain::Tags::Element<Dim>>(box),
            db::get<::Tags::Next<::Tags::TimeStepId>>(box),
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
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \brief Initialize/update items related to mortars after an AMR change
///
/// Mutates:
///   - Tags::MortarData<dim>
///   - Tags::MortarMesh<dim>
///   - Tags::MortarSize<dim>
///   - Tags::MortarNextTemporalId<dim>
///   - evolution::dg::Tags::NormalCovectorAndMagnitude<dim>
///   - Tags::MortarDataHistory<dim, typename dt_variables_tag::type>>
///
/// For p-refinement:
///   - Does nothing to MortarDataHistory (only valid for global time-stepping)
///     or MortarNextTemporalId (only valid for no h-refinement)
///   - Sets the other Mortar tags to be default initialized for each neighbor
///   - Sets the NormalCovectorAndMagnitude to std::nullopt
template <typename Metavariables>
struct ProjectMortars : tt::ConformsTo<amr::protocols::Projector> {
  static constexpr size_t dim = Metavariables::volume_dim;
  using dt_variables_tag = typename db::add_tag_prefix<
      ::Tags::dt, typename Metavariables::system::variables_tag>;
  using mortar_data_history_type =
      typename Tags::MortarDataHistory<dim,
                                       typename dt_variables_tag::type>::type;

  using return_tags =
      tmpl::list<Tags::MortarData<dim>, Tags::MortarMesh<dim>,
                 Tags::MortarSize<dim>, Tags::MortarNextTemporalId<dim>,
                 evolution::dg::Tags::NormalCovectorAndMagnitude<dim>,
                 Tags::MortarDataHistory<dim, typename dt_variables_tag::type>>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<dim>, domain::Tags::Element<dim>,
                 amr::Tags::NeighborInfo<dim>>;

  static void apply(
      const gsl::not_null<::dg::MortarMap<dim, evolution::dg::MortarData<dim>>*>
          mortar_data,
      const gsl::not_null<::dg::MortarMap<dim, Mesh<dim - 1>>*> mortar_mesh,
      const gsl::not_null<
          ::dg::MortarMap<dim, std::array<Spectral::MortarSize, dim - 1>>*>
          mortar_size,
      const gsl::not_null<
          ::dg::MortarMap<dim, TimeStepId>*> /*mortar_next_temporal_id*/,
      const gsl::not_null<
          DirectionMap<dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<dim>>>>>*>
          normal_covector_and_magnitude,
      const gsl::not_null<mortar_data_history_type*>
      /*mortar_data_history*/,
      const Mesh<dim>& new_mesh, const Element<dim>& new_element,
      const std::unordered_map<ElementId<dim>, amr::Info<dim>>& neighbor_info,
      const std::pair<Mesh<dim>, Element<dim>>& /*old_mesh_and_element*/) {
    if (Metavariables::local_time_stepping) {
      ERROR("AMR with local time-stepping is not yet supported");
    }

    mortar_data->clear();
    mortar_mesh->clear();
    mortar_size->clear();
    // mortar_next_temporal_id is not changed, but this will break when
    // h-refinement is enabled and the neighbors are no longer the same
    for (const auto& [direction, neighbors] : new_element.neighbors()) {
      (*normal_covector_and_magnitude)[direction] = std::nullopt;
      for (const auto& neighbor : neighbors) {
        const DirectionalId<dim> mortar_id{direction, neighbor};
        mortar_data->emplace(mortar_id, MortarData<dim>{1});
        const auto new_neighbor_mesh = neighbors.orientation().inverse_map()(
            neighbor_info.at(neighbor).new_mesh);
        mortar_mesh->emplace(
            mortar_id,
            ::dg::mortar_mesh(
                new_mesh.slice_away(direction.dimension()),
                new_neighbor_mesh.slice_away(direction.dimension())));
        mortar_size->emplace(
            mortar_id,
            ::dg::mortar_size(new_element.id(), neighbor, direction.dimension(),
                              neighbors.orientation()));
      }
    }
    for (const auto& direction : new_element.external_boundaries()) {
      (*normal_covector_and_magnitude)[direction] = std::nullopt;
    }
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<::dg::MortarMap<dim, evolution::dg::MortarData<dim>>*>
      /*mortar_data*/,
      const gsl::not_null<::dg::MortarMap<dim, Mesh<dim - 1>>*> /*mortar_mesh*/,
      const gsl::not_null<
          ::dg::MortarMap<dim, std::array<Spectral::MortarSize, dim - 1>>*>
      /*mortar_size*/,
      const gsl::not_null<
          ::dg::MortarMap<dim, TimeStepId>*> /*mortar_next_temporal_id*/,
      const gsl::not_null<
          DirectionMap<dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<dim>>>>>*>
      /*normal_covector_and_magnitude*/,
      const gsl::not_null<mortar_data_history_type*>
      /*mortar_data_history*/,
      const Mesh<dim>& /*new_mesh*/, const Element<dim>& /*new_element*/,
      const std::unordered_map<ElementId<dim>,
                               amr::Info<dim>>& /*neighbor_info*/,
      const tuples::TaggedTuple<Tags...>& /*parent_items*/) {
    ERROR("h-refinement not implemented yet");
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<::dg::MortarMap<dim, evolution::dg::MortarData<dim>>*>
      /*mortar_data*/,
      const gsl::not_null<::dg::MortarMap<dim, Mesh<dim - 1>>*> /*mortar_mesh*/,
      const gsl::not_null<
          ::dg::MortarMap<dim, std::array<Spectral::MortarSize, dim - 1>>*>
      /*mortar_size*/,
      const gsl::not_null<
          ::dg::MortarMap<dim, TimeStepId>*> /*mortar_next_temporal_id*/,
      const gsl::not_null<
          DirectionMap<dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<dim>>>>>*>
      /*normal_covector_and_magnitude*/,
      const gsl::not_null<mortar_data_history_type*>
      /*mortar_data_history*/,
      const Mesh<dim>& /*new_mesh*/, const Element<dim>& /*new_element*/,
      const std::unordered_map<ElementId<dim>,
                               amr::Info<dim>>& /*neighbor_info*/,
      const std::unordered_map<ElementId<dim>, tuples::TaggedTuple<Tags...>>&
      /*children_items*/) {
    ERROR("h-refinement not implemented yet");
  }
};
}  // namespace evolution::dg::Initialization
