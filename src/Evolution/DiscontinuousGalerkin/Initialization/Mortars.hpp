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
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/TrimMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/LtsMode.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Domain;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Spectral {
enum class Quadrature : uint8_t;
}  // namespace Spectral
namespace Tags {
struct LtsMode;
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
::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>> empty_mortar_data(
    const Element<Dim>& element);

template <size_t Dim>
::dg::MortarMap<Dim, MortarInfo<Dim>> mortar_infos(
    const Domain<Dim>& domain, const Element<Dim>& element,
    const Mesh<Dim>& volume_mesh,
    const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh, LtsMode lts_mode);

template <size_t Dim>
std::tuple<::dg::MortarMap<Dim, Mesh<Dim - 1>>,
           ::dg::MortarMap<Dim, TimeStepId>,
           DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                 evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<Dim>>>>>>
mortars_apply_impl(const Element<Dim>& element,
                   const TimeStepId& next_temporal_id,
                   const Mesh<Dim>& volume_mesh,
                   const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh);

template <size_t Dim>
void h_refine_structure(
    gsl::not_null<::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>*>
        mortar_data,
    gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_mesh,
    gsl::not_null<::dg::MortarMap<Dim, MortarInfo<Dim>>*> mortar_infos,
    gsl::not_null<::dg::MortarMap<Dim, TimeStepId>*> mortar_next_temporal_id,
    gsl::not_null<
        DirectionMap<Dim, std::optional<::Variables<tmpl::list<
                              ::evolution::dg::Tags::MagnitudeOfNormal,
                              ::evolution::dg::Tags::NormalCovector<Dim>>>>>*>
        normal_covector_and_magnitude,
    const Domain<Dim>& domain, const Mesh<Dim>& new_mesh,
    const Element<Dim>& new_element,
    const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh,
    const TimeStepId& current_temporal_id, LtsMode lts_mode);
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
 * - Removes: nothing
 * - Modifies: nothing
 */
template <size_t Dim>
struct Mortars {
 public:
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  using simple_tags_from_options = tmpl::list<>;

  using simple_tags =
      tmpl::list<Tags::MortarData<Dim>, Tags::MortarMesh<Dim>,
                 Tags::MortarInfo<Dim>, Tags::MortarNextTemporalId<Dim>,
                 evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
                 Tags::MortarDataHistory<Dim>>;
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
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const auto& element = db::get<::domain::Tags::Element<Dim>>(box);
    const auto& volume_mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& neighbor_mesh = db::get<domain::Tags::NeighborMesh<Dim>>(box);
    const auto lts_mode = db::get<::Tags::LtsMode>(box);
    auto mortar_data = detail::empty_mortar_data(element);
    auto mortar_infos = detail::mortar_infos(domain, element, volume_mesh,
                                             neighbor_mesh, lts_mode);
    auto [mortar_meshes, mortar_next_temporal_ids, normal_covector_quantities] =
        detail::mortars_apply_impl(
            element, db::get<::Tags::Next<::Tags::TimeStepId>>(box),
            db::get<::domain::Tags::Mesh<Dim>>(box),
            db::get<::domain::Tags::NeighborMesh<Dim>>(box));
    typename Tags::MortarDataHistory<Dim>::type boundary_data_history{};
    for (const auto& mortar_id_and_data : mortar_data) {
      if (mortar_infos.at(mortar_id_and_data.first).time_stepping_policy() ==
          TimeSteppingPolicy::Conservative) {
        // default initialize data
        boundary_data_history[mortar_id_and_data.first];
      }
    }
    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(mortar_data), std::move(mortar_meshes),
        std::move(mortar_infos), std::move(mortar_next_temporal_ids),
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
///   - Tags::MortarInfo<dim>
///   - Tags::MortarNextTemporalId<dim>
///   - evolution::dg::Tags::NormalCovectorAndMagnitude<dim>
///   - Tags::MortarDataHistory<dim>>
///
/// For p-refined interfaces:
///   - Regenerates MortarData and MortarInfo (should have no effect)
///   - Sets the NormalCovectorAndMagnitude to std::nullopt if the face mesh
///     changed
///   - Projects the local geometric data (but not the data on the mortar-mesh)
///     in the MortarDataHistory, if present
///   - Does nothing to MortarMesh and MortarNextTemporalId
///
/// For h-refined interfaces:
///   - Regenerates MortarData and MortarInfo
///   - Sets the NormalCovectorAndMagnitude to std::nullopt
///   - Calculates MortarMesh
///   - Sets MortarNextTemporalId to the current temporal id
///   - For local time-stepping:
///     - Removes MortarDataHistory data corresponding to split or joined
///       elements
///     - Projects MortarDataHistory data corresponding to non-h-refined
///       elements onto refined mortars (both geometric and mortar-mesh data)
///     - Creates empty histories for new mortars between two h-refined
///       elements
template <size_t Dim>
struct ProjectMortars : tt::ConformsTo<amr::protocols::Projector> {
 private:
  using magnitude_and_normal_type =
      ::Variables<tmpl::list<::evolution::dg::Tags::MagnitudeOfNormal,
                             ::evolution::dg::Tags::NormalCovector<Dim>>>;

 public:
  using mortar_data_history_tag = Tags::MortarDataHistory<Dim>;
  using mortar_data_history_type = typename mortar_data_history_tag::type;

  using return_tags =
      tmpl::list<Tags::MortarData<Dim>, Tags::MortarMesh<Dim>,
                 Tags::MortarInfo<Dim>, Tags::MortarNextTemporalId<Dim>,
                 evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
                 Tags::MortarDataHistory<Dim>>;
  using argument_tags =
      tmpl::list<domain::Tags::Domain<Dim>, domain::Tags::Mesh<Dim>,
                 domain::Tags::Element<Dim>, domain::Tags::NeighborMesh<Dim>,
                 ::Tags::TimeStepId, ::Tags::LtsMode>;

  static void apply(
      gsl::not_null<::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>*>
          mortar_data,
      gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_mesh,
      gsl::not_null<::dg::MortarMap<Dim, MortarInfo<Dim>>*> mortar_infos,
      gsl::not_null<::dg::MortarMap<Dim, TimeStepId>*> mortar_next_temporal_id,
      gsl::not_null<
          DirectionMap<Dim, std::optional<magnitude_and_normal_type>>*>
          normal_covector_and_magnitude,
      gsl::not_null<mortar_data_history_type*> mortar_data_history,
      const Domain<Dim>& domain, const Mesh<Dim>& new_mesh,
      const Element<Dim>& new_element,
      const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh,
      const TimeStepId& current_temporal_id, LtsMode lts_mode,
      const std::pair<Mesh<Dim>, Element<Dim>>& old_mesh_and_element);

  template <typename... ParentTags>
  static void apply(
      const gsl::not_null<
          ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>*>
          mortar_data,
      const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_mesh,
      const gsl::not_null<::dg::MortarMap<Dim, MortarInfo<Dim>>*> mortar_infos,
      const gsl::not_null<::dg::MortarMap<Dim, TimeStepId>*>
          mortar_next_temporal_id,
      const gsl::not_null<
          DirectionMap<Dim, std::optional<magnitude_and_normal_type>>*>
          normal_covector_and_magnitude,
      const gsl::not_null<mortar_data_history_type*> mortar_data_history,
      const Domain<Dim>& domain, const Mesh<Dim>& new_mesh,
      const Element<Dim>& new_element,
      const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh,
      const TimeStepId& /*possibly_unset*/, const LtsMode lts_mode,
      const tuples::TaggedTuple<ParentTags...>& parent_items) {
    detail::h_refine_structure(
        mortar_data, mortar_mesh, mortar_infos, mortar_next_temporal_id,
        normal_covector_and_magnitude, domain, new_mesh, new_element,
        neighbor_mesh, get<::Tags::TimeStepId>(parent_items), lts_mode);

    const auto& old_element = get<domain::Tags::Element<Dim>>(parent_items);
    const auto& old_histories = get<mortar_data_history_tag>(parent_items);
    for (const auto& [direction, neighbors] : new_element.neighbors()) {
      for (const auto& neighbor : neighbors) {
        const DirectionalId<Dim> mortar_id{direction, neighbor};
        if (mortar_infos->at(mortar_id).time_stepping_policy() !=
            TimeSteppingPolicy::Conservative) {
          continue;
        }
        if (const auto old_history = old_histories.find(mortar_id);
            old_history != old_histories.end()) {
          // The neighbor did not h-refine, so we have to project
          // its mortar data from our parent.
          auto& new_history =
              mortar_data_history->emplace(mortar_id, old_history->second)
                  .first->second;
          new_history.local().clear();
          auto remote_history = new_history.remote();
          const auto& new_mortar_mesh = mortar_mesh->at(mortar_id);
          const auto& orientation = neighbors.orientation(neighbor);
          const auto new_mortar_size = domain::child_size(
              ::dg::mortar_segments(new_element.id(), neighbor,
                                    direction.dimension(), orientation),
              ::dg::mortar_segments(old_element.id(), neighbor,
                                    direction.dimension(), orientation));
          const auto project_mortar_data =
              [&new_mortar_mesh, &new_mortar_size](
                  const TimeStepId& /* id */,
                  const gsl::not_null<::evolution::dg::MortarData<Dim>*> data) {
                const auto& old_mortar_mesh = data->mortar_mesh.value();
                DataVector& vars = data->mortar_data.value();
                vars = Spectral::project(
                    vars, old_mortar_mesh, new_mortar_mesh,
                    make_array<Dim - 1>(Spectral::SegmentSize::Full),
                    new_mortar_size);
                data->mortar_mesh = new_mortar_mesh;
                return true;
              };
          remote_history.for_each(project_mortar_data);
        } else {
          // Neither this element nor the neighbor existed before
          // refinement.
          mortar_data_history->emplace(
              mortar_id, typename mortar_data_history_type::mapped_type{});
        }
      }
    }
  }

  template <typename... ChildTags>
  static void apply(
      const gsl::not_null<
          ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>*>
          mortar_data,
      const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_mesh,
      const gsl::not_null<::dg::MortarMap<Dim, MortarInfo<Dim>>*> mortar_infos,
      const gsl::not_null<::dg::MortarMap<Dim, TimeStepId>*>
          mortar_next_temporal_id,
      const gsl::not_null<
          DirectionMap<Dim, std::optional<magnitude_and_normal_type>>*>
          normal_covector_and_magnitude,
      const gsl::not_null<mortar_data_history_type*> mortar_data_history,
      const Domain<Dim>& domain, const Mesh<Dim>& new_mesh,
      const Element<Dim>& new_element,
      const ::dg::MortarMap<Dim, Mesh<Dim>>& neighbor_mesh,
      const TimeStepId& /*possibly_unset*/, const LtsMode lts_mode,
      const std::unordered_map<
          ElementId<Dim>, tuples::TaggedTuple<ChildTags...>>& children_items) {
    detail::h_refine_structure(
        mortar_data, mortar_mesh, mortar_infos, mortar_next_temporal_id,
        normal_covector_and_magnitude, domain, new_mesh, new_element,
        neighbor_mesh, get<::Tags::TimeStepId>(children_items.begin()->second),
        lts_mode);

    for (const auto& [direction, neighbors] : new_element.neighbors()) {
      for (const auto& neighbor : neighbors) {
        const DirectionalId<Dim> mortar_id{direction, neighbor};
        if (mortar_infos->at(mortar_id).time_stepping_policy() !=
            TimeSteppingPolicy::Conservative) {
          continue;
        }
        std::optional<typename mortar_data_history_type::mapped_type>
            new_history{};
        for (const auto& [child, child_items] : children_items) {
          const auto& old_histories =
              get<mortar_data_history_tag>(child_items);
          if (const auto old_history = old_histories.find(mortar_id);
              old_history != old_histories.end()) {
            // The neighbor did not h-refine, so we have to project
            // its mortar data from our children.
            const auto& new_mortar_mesh = mortar_mesh->at(mortar_id);
            const auto& orientation = neighbors.orientation(neighbor);
            const auto old_mortar_size = domain::child_size(
                ::dg::mortar_segments(child, neighbor, direction.dimension(),
                                      orientation),
                ::dg::mortar_segments(new_element.id(), neighbor,
                                      direction.dimension(), orientation));
            if (not new_history.has_value()) {
              new_history.emplace(old_history->second);
              new_history->local().clear();
              auto remote_history = new_history->remote();
              const auto project_mortar_data =
                  [&new_mortar_mesh, &old_mortar_size](
                      const TimeStepId& /* id */,
                      const gsl::not_null<::evolution::dg::MortarData<Dim>*>
                          data) {
                    const auto& old_mortar_mesh = data->mortar_mesh.value();
                    DataVector& vars = data->mortar_data.value();
                    vars = Spectral::project(
                        vars, old_mortar_mesh, new_mortar_mesh, old_mortar_size,
                        make_array<Dim - 1>(Spectral::SegmentSize::Full));
                    data->mortar_mesh = new_mortar_mesh;
                    return true;
                  };
              remote_history.for_each(project_mortar_data);
            } else {
              auto remote_history = new_history->remote();
              const auto old_remote_history = old_history->second.remote();
              const auto project_mortar_data =
                  [&new_mortar_mesh, &old_mortar_size, &old_remote_history](
                      const TimeStepId& id,
                      const gsl::not_null<::evolution::dg::MortarData<Dim>*>
                          data) {
                    const auto& old_data = old_remote_history.data(id);
                    const auto& old_mortar_mesh =
                        old_data.mortar_mesh.value();
                    data->mortar_data.value() += Spectral::project(
                        old_data.mortar_data.value(), old_mortar_mesh,
                        new_mortar_mesh, old_mortar_size,
                        make_array<Dim - 1>(Spectral::SegmentSize::Full));
                    return true;
                  };
              remote_history.for_each(project_mortar_data);
            }
          }
        }

        if (new_history.has_value()) {
          mortar_data_history->emplace(mortar_id, std::move(*new_history));
        } else {
          // Neither this element nor the neighbor existed before
          // refinement.
          mortar_data_history->emplace(
              mortar_id, typename mortar_data_history_type::mapped_type{});
        }
      }
    }
  }
};
}  // namespace evolution::dg::Initialization
