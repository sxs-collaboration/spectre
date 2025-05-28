// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Spectral {
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
std::tuple<DirectionalIdMap<Dim, evolution::dg::MortarDataHolder<Dim>>,
           DirectionalIdMap<Dim, Mesh<Dim - 1>>,
           DirectionalIdMap<Dim, MortarInfo<Dim>>,
           DirectionalIdMap<Dim, TimeStepId>,
           DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                 evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<Dim>>>>>>
mortars_apply_impl(const Element<Dim>& element,
                   const TimeStepId& next_temporal_id,
                   const Mesh<Dim>& volume_mesh,
                   const DirectionalIdMap<Dim, Mesh<Dim>>& neighbor_mesh);

template <size_t Dim, typename MortarDataHistoryType>
void p_project(
    const gsl::not_null<
        ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>*>
    /* mortar_data */,
    const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_mesh,
    const gsl::not_null<::dg::MortarMap<Dim, MortarInfo<Dim>>*>
    /* mortar_infos */,
    const gsl::not_null<::dg::MortarMap<Dim, TimeStepId>*>
    /* mortar_next_temporal_id */,
    const gsl::not_null<
        DirectionMap<Dim, std::optional<Variables<tmpl::list<
                              evolution::dg::Tags::MagnitudeOfNormal,
                              evolution::dg::Tags::NormalCovector<Dim>>>>>*>
        normal_covector_and_magnitude,
    const gsl::not_null<MortarDataHistoryType*> mortar_data_history,
    const Mesh<Dim>& new_mesh, const Element<Dim>& new_element,
    const std::pair<Mesh<Dim>, Element<Dim>>& old_mesh_and_element) {
  const auto& [old_mesh, old_element] = old_mesh_and_element;
  ASSERT(old_element.id() == new_element.id(),
         "p-refinement should not have changed the element id");

  if (old_mesh == new_mesh) {
    return;
  }

  for (const auto& [direction, neighbors] : new_element.neighbors()) {
    const auto sliced_away_dimension = direction.dimension();
    const auto old_face_mesh = old_mesh.slice_away(sliced_away_dimension);
    const auto new_face_mesh = new_mesh.slice_away(sliced_away_dimension);
    const bool face_mesh_changed = old_face_mesh != new_face_mesh;
    if (face_mesh_changed) {
      (*normal_covector_and_magnitude)[direction] = std::nullopt;
    }
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      if (mortar_mesh->contains(mortar_id)) {
        // mortar_data does not need projecting as it has already been used
        // and will be resized automatically
        // mortar_size does not change as the mortar has not changed
        // next_temporal_id does not change as the mortar has not changed
        if (not mortar_data_history->empty()) {
          auto& boundary_history = mortar_data_history->at(mortar_id);
          auto local_history = boundary_history.local();
          const auto project_local_boundary_data =
              [&new_face_mesh, &new_mesh](
                  const TimeStepId& /* id */,
                  const gsl::not_null<::evolution::dg::MortarData<Dim>*>
                      mortar_data) {
                p_project_geometric_data(mortar_data, new_face_mesh, new_mesh);
              };
          local_history.for_each(project_local_boundary_data);
          boundary_history.clear_coupling_cache();
        }
      } else {
        ERROR("h-refinement not implemented yet");
      }
    }
  }

  for (const auto& direction : new_element.external_boundaries()) {
    const auto sliced_away_dimension = direction.dimension();
    const auto old_face_mesh = old_mesh.slice_away(sliced_away_dimension);
    const auto new_face_mesh = new_mesh.slice_away(sliced_away_dimension);
    const bool face_mesh_changed = old_face_mesh != new_face_mesh;
    if (face_mesh_changed) {
      (*normal_covector_and_magnitude)[direction] = std::nullopt;
    }
  }
}
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
 *   - `Tags::MortarInfo<Dim>`
 *   - `Tags::MortarNextTemporalId<Dim>`
 *   - `evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>`
 *   - `evolution::dg::Tags::BoundaryData<Dim>`
 * - Removes: nothing
 * - Modifies: nothing
 */
template <size_t Dim, typename System>
struct Mortars {
  template <typename MappedType>
  using MortarMap = DirectionalIdMap<Dim, MappedType>;

 public:
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;

  using simple_tags = tmpl::list<
      Tags::MortarData<Dim>, Tags::MortarMesh<Dim>, Tags::MortarInfo<Dim>,
      Tags::MortarNextTemporalId<Dim>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
      Tags::MortarDataHistory<
          Dim, typename db::add_tag_prefix<
                   ::Tags::dt, typename System::variables_tag>::type>,
      evolution::dg::Tags::BoundaryData<Dim>>;
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
    auto [mortar_data, mortar_meshes, mortar_infos, mortar_next_temporal_ids,
          normal_covector_quantities] =
        detail::mortars_apply_impl(
            db::get<::domain::Tags::Element<Dim>>(box),
            db::get<::Tags::Next<::Tags::TimeStepId>>(box),
            db::get<::domain::Tags::Mesh<Dim>>(box),
            db::get<::domain::Tags::NeighborMesh<Dim>>(box));
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
        std::move(mortar_infos), std::move(mortar_next_temporal_ids),
        std::move(normal_covector_quantities), std::move(boundary_data_history),
        typename evolution::dg::Tags::BoundaryData<Dim>::type{});
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \brief Initialize/update items related to mortars after an AMR change
///
/// Mutates:
///   - Tags::MortarData<dim>
///   - Tags::MortarMesh<dim>
///   - Tags::MortarInfo<dim>
///   - Tags::MortarNextTemporalId<dim>
///   - evolution::dg::Tags::NormalCovectorAndMagnitude<dim>
///   - Tags::MortarDataHistory<dim, typename dt_variables_tag::type>>
///
/// For p-refinement:
///   - Sets the NormalCovectorAndMagnitude to std::nullopt if the face mesh
///     changed
///   - Projects the local geometric data (but not the data on the mortar-mesh)
///     in the MortarDataHistory, if present
///   - Does nothing to the other tags
template <typename Metavariables>
struct ProjectMortars : tt::ConformsTo<amr::protocols::Projector> {
 private:
  using magnitude_and_normal_type = ::Variables<tmpl::list<
      ::evolution::dg::Tags::MagnitudeOfNormal,
      ::evolution::dg::Tags::NormalCovector<Metavariables::volume_dim>>>;

 public:
  static constexpr size_t dim = Metavariables::volume_dim;
  using dt_variables_tag = typename db::add_tag_prefix<
      ::Tags::dt, typename Metavariables::system::variables_tag>;
  using mortar_data_history_type =
      typename Tags::MortarDataHistory<dim,
                                       typename dt_variables_tag::type>::type;

  using return_tags =
      tmpl::list<Tags::MortarData<dim>, Tags::MortarMesh<dim>,
                 Tags::MortarInfo<dim>, Tags::MortarNextTemporalId<dim>,
                 evolution::dg::Tags::NormalCovectorAndMagnitude<dim>,
                 Tags::MortarDataHistory<dim, typename dt_variables_tag::type>>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<dim>, domain::Tags::Element<dim>,
                 domain::Tags::NeighborMesh<dim>>;

  static void apply(
      const gsl::not_null<
          ::dg::MortarMap<dim, evolution::dg::MortarDataHolder<dim>>*>
          mortar_data,
      const gsl::not_null<::dg::MortarMap<dim, Mesh<dim - 1>>*> mortar_mesh,
      const gsl::not_null<::dg::MortarMap<dim, MortarInfo<dim>>*> mortar_infos,
      const gsl::not_null<::dg::MortarMap<dim, TimeStepId>*>
          mortar_next_temporal_id,
      const gsl::not_null<
          DirectionMap<dim, std::optional<magnitude_and_normal_type>>*>
          normal_covector_and_magnitude,
      const gsl::not_null<mortar_data_history_type*> mortar_data_history,
      const Mesh<dim>& new_mesh, const Element<dim>& new_element,
      const ::dg::MortarMap<dim, Mesh<dim>>& /*neighbor_mesh*/,
      const std::pair<Mesh<dim>, Element<dim>>& old_mesh_and_element) {
    detail::p_project(mortar_data, mortar_mesh, mortar_infos,
                      mortar_next_temporal_id, normal_covector_and_magnitude,
                      mortar_data_history, new_mesh, new_element,
                      old_mesh_and_element);
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<
          ::dg::MortarMap<dim, evolution::dg::MortarDataHolder<dim>>*>
      /*mortar_data*/,
      const gsl::not_null<::dg::MortarMap<dim, Mesh<dim - 1>>*> /*mortar_mesh*/,
      const gsl::not_null<::dg::MortarMap<dim, MortarInfo<dim>>*>
      /*mortar_infos*/,
      const gsl::not_null<
          ::dg::MortarMap<dim, TimeStepId>*> /*mortar_next_temporal_id*/,
      const gsl::not_null<
          DirectionMap<dim, std::optional<magnitude_and_normal_type>>*>
      /*normal_covector_and_magnitude*/,
      const gsl::not_null<mortar_data_history_type*>
      /*mortar_data_history*/,
      const Mesh<dim>& /*new_mesh*/, const Element<dim>& /*new_element*/,
      const ::dg::MortarMap<dim, Mesh<dim>>& /*neighbor_mesh*/,
      const tuples::TaggedTuple<Tags...>& /*parent_items*/) {
    ERROR("h-refinement not implemented yet");
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<
          ::dg::MortarMap<dim, evolution::dg::MortarDataHolder<dim>>*>
      /*mortar_data*/,
      const gsl::not_null<::dg::MortarMap<dim, Mesh<dim - 1>>*> /*mortar_mesh*/,
      const gsl::not_null<::dg::MortarMap<dim, MortarInfo<dim>>*>
      /*mortar_infos*/,
      const gsl::not_null<
          ::dg::MortarMap<dim, TimeStepId>*> /*mortar_next_temporal_id*/,
      const gsl::not_null<
          DirectionMap<dim, std::optional<magnitude_and_normal_type>>*>
      /*normal_covector_and_magnitude*/,
      const gsl::not_null<mortar_data_history_type*>
      /*mortar_data_history*/,
      const Mesh<dim>& /*new_mesh*/, const Element<dim>& /*new_element*/,
      const ::dg::MortarMap<dim, Mesh<dim>>& /*neighbor_mesh*/,
      const std::unordered_map<ElementId<dim>, tuples::TaggedTuple<Tags...>>&
      /*children_items*/) {
    ERROR("h-refinement not implemented yet");
  }
};
}  // namespace evolution::dg::Initialization
