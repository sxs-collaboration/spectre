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
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
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

namespace dg {
namespace Actions {

namespace DgActions_detail {
template <size_t volume_dim, typename Tag>
using mortars_tag = Tags::HistoryBoundaryVariables<
    std::pair<Direction<volume_dim>, ElementId<volume_dim>>, Tag,
    boost::hash<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>;
}  // namespace DgActions_detail

template <typename Metavariables>
struct ComputeBoundaryFlux {
  using PackagedData = Variables<
      typename Metavariables::normal_dot_numerical_flux::type::package_tags>;
  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = system::volume_dim;

  struct FluxesTag {
    using temporal_id = TimeId;
    using type = std::unordered_map<
        TimeId, std::unordered_map<
                    std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
                    PackagedData,
                    boost::hash<std::pair<Direction<volume_dim>,
                                          ElementId<volume_dim>>>>>;
  };

  using inbox_tags = tmpl::list<FluxesTag>;

 private:
  template <typename Flux, typename... NumericalFluxTags, typename... Tags,
            typename... ArgumentTags, typename... Args>
  static void apply_normal_dot_numerical_flux(
      const gsl::not_null<Variables<tmpl::list<NumericalFluxTags...>>*>
          numerical_fluxes,
      const Flux& flux,
      const Variables<tmpl::list<Tags...>>& self_packaged_data,
      const Variables<tmpl::list<Tags...>>& neighbor_packaged_data) noexcept {
    flux(make_not_null(&get<NumericalFluxTags>(*numerical_fluxes))...,
         get<Tags>(self_packaged_data)...,
         get<Tags>(neighbor_packaged_data)...);
  }

 public:
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using variables_tag = typename system::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;

    using normal_dot_flux_tag =
        db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>;
    using normal_dot_numerical_flux_tag =
        db::add_tag_prefix<Tags::NormalDotNumericalFlux, variables_tag>;
    using mortars_normal_dot_fluxes_tag =
        DgActions_detail::mortars_tag<volume_dim, normal_dot_flux_tag>;
    using mortars_packaged_data_tag = DgActions_detail::mortars_tag<
        volume_dim,
        Tags::Variables<typename Metavariables::normal_dot_numerical_flux::
                            type::package_tags>>;

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);
    const auto& extents = db::get<Tags::Extents<volume_dim>>(box);
    const auto& local_mortar_data = db::get<mortars_packaged_data_tag>(box);
    const auto& local_normal_dot_flux =
        db::get<mortars_normal_dot_fluxes_tag>(box);

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
      const PackagedData& self_packaged_data =
          local_mortar_data.at(std::make_pair(direction, neighbor));
      const PackagedData& neighbor_packaged_data = neighbor_data_it->second;

      // All of this needs to be fixed for hp-adaptivity to handle
      // projections correctly
      const auto& face_normal =
          db::get<Tags::UnnormalizedFaceNormal<volume_dim>>(box).at(direction);

      // Compute numerical flux
      db::item_type<normal_dot_numerical_flux_tag> normal_dot_numerical_fluxes(
          face_normal.begin()->size(), 0.0);
      apply_normal_dot_numerical_flux(
          make_not_null(&normal_dot_numerical_fluxes),
          normal_dot_numerical_flux_computer, self_packaged_data,
          neighbor_packaged_data);

      // Needs fixing for GH/curved
      db::item_type<dt_variables_tag> lifted_data(dg::lift_flux(
          local_normal_dot_flux.at(std::make_pair(direction, neighbor)),
          std::move(normal_dot_numerical_fluxes), extents[dimension],
          magnitude(face_normal)));

      db::mutate<dt_variables_tag>(box, [
        &lifted_data, &extents, &dimension, &direction
      ](db::item_type<dt_variables_tag> & dt_vars) noexcept {
        add_slice_to_data(
            make_not_null(&dt_vars), lifted_data, extents, dimension,
            direction.side() == Side::Lower ? 0 : extents[dimension] - 1);
      });
    }

    return std::make_tuple(
        db::create_from<db::RemoveTags<mortars_normal_dot_fluxes_tag,
                                       mortars_packaged_data_tag>>(
            std::move(box)));
  }

  template <typename DbTags, typename... InboxTags, typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept {
    const auto& time_id = db::get<Tags::TimeId>(box);
    const auto& inbox = tuples::get<FluxesTag>(inboxes);
    auto receives = inbox.find(time_id);
    const auto number_of_neighbors =
        db::get<Tags::Element<volume_dim>>(box).number_of_neighbors();
    const size_t num_receives =
        receives == inbox.end() ? 0 : receives->second.size();
    return num_receives == number_of_neighbors;
  }
};

struct SendDataForFluxes {
 private:
  template <typename PackagedData, typename NumericalFluxComputer,
            typename... NormalDotFluxTags, typename TagsList, typename... Args,
            typename... SliceTags>
  static void get_packaged_data_from_numerical_flux(
      const gsl::not_null<PackagedData*> packaged_data,
      const NumericalFluxComputer& numerical_flux_computer,
      const Variables<tmpl::list<NormalDotFluxTags...>>& boundary_flux,
      const Variables<TagsList>& sliced_variables,
      const tmpl::list<SliceTags...>& /*meta*/, Args&&... args) noexcept {
    numerical_flux_computer.package_data(
        packaged_data, get<NormalDotFluxTags>(boundary_flux)...,
        get<SliceTags>(sliced_variables)..., std::forward<Args>(args)...);
  }

  template <typename Metavariables, typename... NormalDotFluxTags,
            typename... VariablesTags, size_t Dim, typename Frame,
            typename... ArgumentTags>
  static void apply_flux(
      gsl::not_null<Variables<tmpl::list<NormalDotFluxTags...>>*> boundary_flux,
      const Variables<tmpl::list<VariablesTags...>>& boundary_variables,
      const tnsr::i<DataVector, Dim, Frame>& unit_face_normal,
      tmpl::list<ArgumentTags...> /*meta*/) noexcept {
    Metavariables::system::normal_dot_fluxes::apply(
        make_not_null(&get<NormalDotFluxTags>(*boundary_flux))...,
        get<ArgumentTags>(boundary_variables)..., unit_face_normal);
  }

 public:
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
    constexpr size_t volume_dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;
    using PackagedData = Variables<
        typename Metavariables::normal_dot_numerical_flux::type::package_tags>;

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& extents = db::get<Tags::Extents<volume_dim>>(box);
    const auto& time_id = db::get<Tags::TimeId>(box);

    using normal_dot_flux_tag =
        db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>;
    using NormalDotFluxType = db::item_type<normal_dot_flux_tag>;

    std::unordered_map<
        std::pair<Direction<volume_dim>, ElementId<volume_dim>>, PackagedData,
        boost::hash<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>
        mortars_packaged_data;
    std::unordered_map<
        std::pair<Direction<volume_dim>, ElementId<volume_dim>>,
        NormalDotFluxType,
        boost::hash<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>
        mortars_normal_dot_fluxes;

    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_neighbors.second;
      ASSERT(neighbors_in_direction.size() == 1,
             "h-adaptivity is not supported yet.\nDirection: "
                 << direction << "\nDimension: " << dimension
                 << "\nNeighbors:\n"
                 << neighbors_in_direction);
      const auto& orientation = neighbors_in_direction.orientation();
      const auto boundary_extents = extents.slice_away(dimension);
      auto boundary_variables = db::data_on_slice(
          box, extents, dimension,
          direction.side() == Side::Lower ? 0 : extents[dimension] - 1,
          tmpl::remove_duplicates<
              tmpl::append<typename Metavariables::normal_dot_numerical_flux::
                               type::slice_tags,
                           typename Metavariables::system::normal_dot_fluxes::
                               argument_tags>>{});

      // Everything in the loop over neighbors needs to be fixed for
      // hp-adaptivity to handle projections correctly
      for (const auto& neighbor : neighbors_in_direction) {
        const auto& face_normal =
            db::get<Tags::UnnormalizedFaceNormal<volume_dim>>(box).at(
                direction);
        DataVector magnitude_of_face_normal = magnitude(face_normal);
        std::decay_t<decltype(face_normal)> unit_face_normal(
            magnitude_of_face_normal.size(), 0.0);
        for (size_t d = 0; d < volume_dim; ++d) {
          unit_face_normal.get(d) =
              face_normal.get(d) / magnitude_of_face_normal;
        }

        // This needs to be updated for conservative systems where we can just
        // slice the flux from the volume.
        NormalDotFluxType local_boundary_flux(face_normal.begin()->size(), 0.0);
        apply_flux<Metavariables>(
            make_not_null(&local_boundary_flux), boundary_variables,
            unit_face_normal,
            typename system::normal_dot_fluxes::argument_tags{});

        // We compute the parts of the numerical flux that only depend on data
        // from this side of the mortar now, then package it into a Variables.
        // We store one copy of the Variables and send another, since we need
        // the data on both sides of the mortar.
        PackagedData packaged_data(face_normal.begin()->size(), 0.0);
        get_packaged_data_from_numerical_flux(
            make_not_null(&packaged_data), normal_dot_numerical_flux_computer,
            local_boundary_flux, boundary_variables,
            typename Metavariables::normal_dot_numerical_flux::type::
                slice_tags{},
            unit_face_normal);

        const auto direction_from_neighbor = orientation(direction.opposite());

        // orient_variables_on_slice only needs to be done in the case where
        // the data is oriented differently. This needs to improved later.
        // Note: avoiding the same-orientation-on-both-sides copy is possible
        // even with AMR since the quantities are already on the mortar at this
        // point
        auto neighbor_packaged_data = orient_variables_on_slice(
            packaged_data, boundary_extents, dimension, orientation);

        receiver_proxy[neighbor]
            .template receive_data<
                typename ComputeBoundaryFlux<Metavariables>::FluxesTag>(
                time_id, std::make_pair(std::make_pair(direction_from_neighbor,
                                                       element.id()),
                                        std::move(neighbor_packaged_data)));

        mortars_packaged_data.emplace(std::make_pair(direction, neighbor),
                                      std::move(packaged_data));
        mortars_normal_dot_fluxes.emplace(std::make_pair(direction, neighbor),
                                          std::move(local_boundary_flux));
      }  // loop over neighbors_in_direction
    }    // loop over element.neighbors()

    using mortars_normal_dot_fluxes_tag =
        DgActions_detail::mortars_tag<volume_dim, normal_dot_flux_tag>;
    using mortars_packaged_data_tag = DgActions_detail::mortars_tag<
        volume_dim,
        Tags::Variables<typename Metavariables::normal_dot_numerical_flux::
                            type::package_tags>>;
    return std::make_tuple(
        db::create_from<db::RemoveTags<>,
                        db::AddTags<mortars_packaged_data_tag,
                                    mortars_normal_dot_fluxes_tag>>(
            box, std::move(mortars_packaged_data),
            std::move(mortars_normal_dot_fluxes)));
  }
};
}  // namespace Actions
}  // namespace dg
