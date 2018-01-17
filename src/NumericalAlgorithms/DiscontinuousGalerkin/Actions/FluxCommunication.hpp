// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines actions ComputeBoundaryFlux and SendDataForFluxes

#pragma once

#include <boost/functional/hash.hpp>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Receive boundary data from neighbors and compute boundary
/// contribution to the time derivative.
///
/// \note The data must be oriented by the sender, but this class
/// handles projection, etc.
///
/// Uses:
/// - ConstGlobalCache: Metavariables::numerical_flux
/// - DataBox: Tags::Element<volume_dim>,
///   Tags::HistoryBoundaryVariables<Direction<volume_dim>,
///       system::variables_tag>, Tags::Extents<volume_dim>, Tags::TimeId,
///   Tags::UnnormalizedFaceNormal<volume_dim>
///
/// DataBox changes:
/// - Adds: system::dt_variables_tag
/// - Removes: Tags::HistoryBoundaryVariables<Direction<volume_dim>,
///       system::variables_tag>
/// - Modifies: nothing
template <typename System>
struct ComputeBoundaryFlux {
  struct FluxesTag {
    using temporal_id = TimeId;
    using type =
        std::unordered_map<
            TimeId,
            std::unordered_map<
                std::pair<Direction<System::volume_dim>,
                          ElementId<System::volume_dim>>,
                db::item_type<typename System::variables_tag>,
                boost::hash<std::pair<Direction<System::volume_dim>,
                                      ElementId<System::volume_dim>>>>>;
  };

  using inbox_tags = tmpl::list<FluxesTag>;

  template <typename Flux, typename... Tags, typename... ArgumentTags>
  static auto apply_flux(const Flux& flux,
                         const tuples::TaggedTuple<Tags...>& self_data,
                         const tuples::TaggedTuple<Tags...>& neighbor_data,
                         tmpl::list<ArgumentTags...> /*meta*/) noexcept {
    return flux(tuples::get<ArgumentTags>(self_data)...,
                tuples::get<ArgumentTags>(neighbor_data)...);
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(cpp17::is_same_v<System, typename Metavariables::system>,
                  "Inconsistent systems");
    constexpr const size_t volume_dim = System::volume_dim;
    using variables_tag = typename System::variables_tag;
    using dt_variables_tag = typename System::dt_variables_tag;

    const auto& flux_computer =
        get<typename Metavariables::numerical_flux>(cache);
    const auto& extents = db::get<Tags::Extents<volume_dim>>(box);
    const auto& local_data =
        db::get<Tags::HistoryBoundaryVariables<
                  Direction<volume_dim>, variables_tag>>(box);

    const auto& element = db::get<Tags::Element<volume_dim>>(box);

    auto& inbox = tuples::get<FluxesTag>(inboxes);
    const auto& time_id = db::get<Tags::TimeId>(box);
    auto remote_data = std::move(inbox[time_id]);
    inbox.erase(time_id);

    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_neighbors.second;
      ASSERT(neighbors_in_direction.size() == 1,
             "Complex mortars unimplemented");
      const auto& neighbor = *neighbors_in_direction.begin();
      const auto neighbor_data_it =
          remote_data.find(std::make_pair(direction, neighbor));
      ASSERT(remote_data.end() != neighbor_data_it,
             "Attempting to access neighbor flux data that does not exist. "
             "The neighbor attempting to be accessed is: "
             << neighbor << " in direction " << direction
             << ". The known keys are " << keys_of(remote_data) << ".");
      const auto& local_value = local_data.at(direction);
      const auto& neighbor_value = neighbor_data_it->second;

      // This needs to be fixed for hp-adaptivity to handle
      // projections correctly

      const auto& face_normal =
          db::get<Tags::UnnormalizedFaceNormal<volume_dim>>(box).at(direction);

      DataVector magnitude_of_face_normal = magnitude(face_normal);

      std::decay_t<decltype(face_normal)> unit_face_normal(
          magnitude_of_face_normal.size(), 0.0);
      for (size_t d = 0; d < volume_dim; ++d) {
        unit_face_normal.get(d) = face_normal.get(d) / magnitude_of_face_normal;
      }

      // Using this instead of auto prevents incomprehensible errors
      // if the return type of compute_flux is wrong.
      using FluxType = db::item_type<
          db::add_tag_prefix<Tags::Flux, variables_tag,
                             tmpl::size_t<volume_dim>, Frame::Inertial>>;
      const FluxType local_boundary_flux =
          System::compute_flux::apply(local_value);
      const FluxType neighbor_boundary_flux =
          System::compute_flux::apply(neighbor_value);
      using normal_flux_tag =
          db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>;
      db::item_type<normal_flux_tag> local_normal_flux(
          local_boundary_flux.number_of_grid_points(), 0.);
      db::item_type<normal_flux_tag> neighbor_normal_flux(
          local_boundary_flux.number_of_grid_points(), 0.);
      tmpl::for_each<typename variables_tag::tags_list>([
        &local_normal_flux, &neighbor_normal_flux, &local_boundary_flux,
        &neighbor_boundary_flux, &unit_face_normal
      ](auto tag) noexcept {
        using Tag = tmpl::type_from<decltype(tag)>;
        using flux_tag = Tags::Flux<Tag, tmpl::size_t<2>, Frame::Inertial>;

        const auto& local_bf = get<flux_tag>(local_boundary_flux);
        auto& local_nf = get<Tags::NormalDotFlux<Tag>>(local_normal_flux);
        const auto& neighbor_bf = get<flux_tag>(neighbor_boundary_flux);
        auto& neighbor_nf = get<Tags::NormalDotFlux<Tag>>(neighbor_normal_flux);
        for (auto flux_it = local_bf.begin(); flux_it != local_bf.end();
             ++flux_it) {
          const auto flux_index =
              local_bf.get_tensor_index(flux_it);
          const size_t contract_index = flux_index[0];
          const auto other_indices =
              all_but_specified_element_of<0>(flux_index);
          local_nf.get(other_indices) +=
              unit_face_normal.get(contract_index) * local_bf.get(flux_index);
          neighbor_nf.get(other_indices) +=
              unit_face_normal.get(contract_index) *
              neighbor_bf.get(flux_index);
        }
      });

      const tuples::TaggedTuple<variables_tag, normal_flux_tag> self_data(
          local_value, std::move(local_normal_flux));
      const tuples::TaggedTuple<variables_tag, normal_flux_tag> neighbor_data(
          neighbor_value, std::move(neighbor_normal_flux));

      auto normal_dot_numerical_flux = apply_flux(
          flux_computer, self_data, neighbor_data,
          typename std::decay_t<decltype(flux_computer)>::argument_tags{});

      // Needs fixing for GH/curved
      auto lifted_data(dg::lift_flux(tuples::get<normal_flux_tag>(self_data),
                                     std::move(normal_dot_numerical_flux),
                                     extents[dimension],
                                     std::move(magnitude_of_face_normal)));

      db::mutate<dt_variables_tag>(box, [
        &lifted_data, &extents, &dimension, &direction
      ](db::item_type<dt_variables_tag> & dt_vars) noexcept {
        add_slice_to_data(
            make_not_null(&dt_vars), lifted_data, extents, dimension,
            direction.side() == Side::Lower ? 0 : extents[dimension] - 1);
      });
    }

    return std::make_tuple(
        db::create_from<db::RemoveTags<Tags::HistoryBoundaryVariables<
            Direction<volume_dim>, variables_tag>>>(box));
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    const auto& time_id = db::get<Tags::TimeId>(box);
    const auto& inbox = tuples::get<FluxesTag>(inboxes);
    auto receives = inbox.find(time_id);
    const auto number_of_neighbors =
        db::get<Tags::Element<Metavariables::system::volume_dim>>(box)
            .number_of_neighbors();
    const size_t num_receives =
        receives == inbox.end() ? 0 : receives->second.size();
    return num_receives == number_of_neighbors;
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Compute local boundary data needed for fluxes and send it
/// to neighbors.
///
/// \note The data is oriented for the receiver, but the receiver is
/// responsible for projection, etc.
///
/// Uses:
/// - ConstGlobalCache: Receiver
/// - DataBox: system::variables_tag, Tags::Element<volume_dim>,
///   Tags::Extents<volume_dim>, Tags::TimeId
///
/// DataBox changes:
/// - Adds: Tags::HistoryBoundaryVariables<Direction<volume_dim>,
///       system::variables_tag>
/// - Removes: nothing
/// - Modifies: nothing
struct SendDataForFluxes {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    constexpr const size_t volume_dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& extents = db::get<Tags::Extents<volume_dim>>(box);
    const auto& time_id = db::get<Tags::TimeId>(box);
    const auto& evolved_vars = db::get<variables_tag>(box);

    std::unordered_map<Direction<volume_dim>, db::item_type<variables_tag>>
        mortars;

    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const auto& neighbors_in_direction = direction_neighbors.second;

      const size_t dimension = direction.dimension();
      const auto& segment_id = gsl::at(element.id().segment_ids(), dimension);
      const auto& orientation = neighbors_in_direction.orientation();
      const auto boundary_extents = extents.slice_away(dimension);
      auto boundary_variables = data_on_slice(
          evolved_vars, extents, dimension,
          direction.side() == Side::Lower ? 0 : extents[dimension] - 1);

      for (const auto& neighbor : neighbors_in_direction) {
        const auto& neighbor_segment_id =
            gsl::at(neighbor.segment_ids(), dimension);

        const bool neighbor_is_on_opposite_side_of_mortar =
            element.id().block_id() != neighbor.block_id() or
            not segment_id.overlaps(neighbor_segment_id);
        const auto direction_from_neighbor =
            neighbor_is_on_opposite_side_of_mortar
                ? orientation(direction.opposite())
                : direction;
        auto neighbor_variables =
            neighbor_is_on_opposite_side_of_mortar
                ? orient_variables_on_slice(boundary_variables,
                                            boundary_extents, dimension,
                                            orientation)
                : boundary_variables;

        receiver_proxy[ElementIndex<volume_dim>(neighbor)]
            .template receive_data<
                typename ComputeBoundaryFlux<system>::FluxesTag>(
                time_id,
                std::make_pair(
                    std::make_pair(direction_from_neighbor, element.id()),
                    std::move(neighbor_variables)));
      }

      mortars.emplace(direction, std::move(boundary_variables));
    }
    return std::make_tuple(
        db::create_from<
            db::RemoveTags<>,
            db::AddTags<Tags::HistoryBoundaryVariables<Direction<volume_dim>,
                                                       variables_tag>>>(
            box, std::move(mortars)));
  }
};
}  // namespace Actions
