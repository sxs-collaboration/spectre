// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines actions ComputeBoundaryFlux and SendDataForFluxes

#pragma once

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
// IWYU pragma: no_include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Direction.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare db::DataBox
namespace Tags {
struct TimeId;
template <typename Tag, typename MagnitudeTag>
struct Normalized;
}  // namespace Tags
/// \endcond

namespace dg {
namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Receive boundary data from neighbors and compute boundary
/// contribution to the time derivative.
///
/// Uses:
/// - ConstGlobalCache: Metavariables::normal_dot_numerical_flux
/// - DataBox:
///   Tags::Element<volume_dim>,
///   Tags::Extents<volume_dim>,
///   Tags::Interface<Tags::InternalDirections<volume_dim>,
///                   typename System::template magnitude_tag<
///                       Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   Tags::Mortars<implementation_defined_local_data>,
///   Tags::TimeId,
///   db::add_tag_prefix<Tags::dt, variables_tag>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: Tags::Mortars<implementation_defined_local_data>
/// - Modifies: db::add_tag_prefix<Tags::dt, variables_tag>
template <typename Metavariables>
struct ComputeBoundaryFlux {
  using PackagedData = Variables<
      typename Metavariables::normal_dot_numerical_flux::type::package_tags>;
  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = system::volume_dim;

  using normal_dot_fluxes_tag =
      db::add_tag_prefix<Tags::NormalDotFlux, typename system::variables_tag>;
  using NormalDotFluxesType = db::item_type<normal_dot_fluxes_tag>;

  using LocalData = Variables<tmpl::remove_duplicates<
      tmpl::append<typename NormalDotFluxesType::tags_list,
                   typename PackagedData::tags_list>>>;

  using mortars_local_data_tag =
      Tags::Mortars<Tags::Variables<typename LocalData::tags_list>, volume_dim>;

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
  template <typename Flux, typename... NumericalFluxTags, typename... SelfTags,
            typename... PackagedTags, typename... ArgumentTags,
            typename... Args>
  static void apply_normal_dot_numerical_flux(
      const gsl::not_null<Variables<tmpl::list<NumericalFluxTags...>>*>
          numerical_fluxes,
      const Flux& flux,
      const Variables<tmpl::list<SelfTags...>>& self_packaged_data,
      const Variables<tmpl::list<PackagedTags...>>&
          neighbor_packaged_data) noexcept {
    flux(make_not_null(&get<NumericalFluxTags>(*numerical_fluxes))...,
         get<PackagedTags>(self_packaged_data)...,
         get<PackagedTags>(neighbor_packaged_data)...);
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

    using normal_dot_numerical_flux_tag =
        db::add_tag_prefix<Tags::NormalDotNumericalFlux, variables_tag>;

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);
    const auto& extents = db::get<Tags::Extents<volume_dim>>(box);
    const auto& local_data = db::get<mortars_local_data_tag>(box);

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
      const auto& local_mortar_data =
          local_data.at(std::make_pair(direction, neighbor));
      const PackagedData& neighbor_mortar_data = neighbor_data_it->second;

      // All of this needs to be fixed for hp-adaptivity to handle
      // projections correctly
      const auto& magnitude_of_face_normal =
          db::get<
              Tags::Interface<Tags::InternalDirections<volume_dim>,
                              typename system::template magnitude_tag<
                                  Tags::UnnormalizedFaceNormal<volume_dim>>>>(
              box)
              .at(direction);

      // Compute numerical flux
      db::item_type<normal_dot_numerical_flux_tag> normal_dot_numerical_fluxes(
          magnitude_of_face_normal.begin()->size(), 0.0);
      apply_normal_dot_numerical_flux(
          make_not_null(&normal_dot_numerical_fluxes),
          normal_dot_numerical_flux_computer, local_mortar_data,
          neighbor_mortar_data);

      db::item_type<dt_variables_tag> lifted_data(dg::lift_flux(
          local_mortar_data, std::move(normal_dot_numerical_fluxes),
          extents[dimension], magnitude_of_face_normal));

      db::mutate<dt_variables_tag>(box, [
        &lifted_data, &extents, &dimension, &direction
      ](db::item_type<dt_variables_tag> & dt_vars) noexcept {
        add_slice_to_data(make_not_null(&dt_vars), lifted_data, extents,
                          dimension, index_to_slice_at(extents, direction));
      });
    }

    return std::make_tuple(
        db::create_from<db::RemoveTags<mortars_local_data_tag>>(
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

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Send local boundary data needed for fluxes to neighbors.
///
/// With:
/// - `Interface<Tag> =
///   Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>`
///
/// Uses:
/// - ConstGlobalCache: Metavariables::normal_dot_numerical_flux
/// - DataBox:
///   Tags::Element<volume_dim>,
///   Interface<Tags listed in
///             Metavariables::normal_dot_numerical_flux::type::slice_tags>,
///   Interface<Tags::Extents<volume_dim - 1>>,
///   Interface<db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>,
///   Interface<typename System::template magnitude_tag<
///             Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   Tags::TimeId
///
/// DataBox changes:
/// - Adds: Tags::Mortars<implementation_defined_local_data>
/// - Removes: Interface<db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>
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
    using ReceiverAction = ComputeBoundaryFlux<Metavariables>;
    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;

    using normal_dot_fluxes_tag =
        typename ReceiverAction::normal_dot_fluxes_tag;

    using interface_normal_dot_fluxes_tag =
        Tags::Interface<Tags::InternalDirections<volume_dim>,
                        normal_dot_fluxes_tag>;

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& time_id = db::get<Tags::TimeId>(box);

    using LocalData = typename ReceiverAction::LocalData;

    std::unordered_map<
        std::pair<Direction<volume_dim>, ElementId<volume_dim>>, LocalData,
        boost::hash<std::pair<Direction<volume_dim>, ElementId<volume_dim>>>>
        mortars_local_data;

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
      const auto& boundary_extents =
          db::get<Tags::Interface<Tags::InternalDirections<volume_dim>,
                                  Tags::Extents<volume_dim - 1>>>(box)
              .at(direction);

      // Everything below here needs to be fixed for
      // hp-adaptivity to handle projections correctly

      // We compute the parts of the numerical flux that only depend on data
      // from this side of the mortar now, then package it into a Variables.
      // We store one copy of the Variables and send another, since we need
      // the data on both sides of the mortar.
      using normal_tag = Tags::UnnormalizedFaceNormal<volume_dim>;
      using package_arguments = tmpl::append<
          typename db::item_type<normal_dot_fluxes_tag>::tags_list,
          typename Metavariables::normal_dot_numerical_flux::type::slice_tags,
          tmpl::list<Tags::Normalized<
              normal_tag,
              typename system::template magnitude_tag<normal_tag>>>>;
      const auto packaged_data = db::apply<tmpl::transform<
          package_arguments,
          tmpl::bind<Tags::Interface, Tags::InternalDirections<volume_dim>,
                     tmpl::_1>>>(
          [&boundary_extents, &direction, &normal_dot_numerical_flux_computer](
              const auto&... args) noexcept {
            typename ReceiverAction::PackagedData ret(
                boundary_extents.product(), 0.0);
            normal_dot_numerical_flux_computer.package_data(
                make_not_null(&ret), args.at(direction)...);
            return ret;
          },
          box);

      LocalData local_data(boundary_extents.product());
      local_data.assign_subset(
          db::get<interface_normal_dot_fluxes_tag>(box).at(direction));
      local_data.assign_subset(packaged_data);

      const auto direction_from_neighbor = orientation(direction.opposite());

      // orient_variables_on_slice only needs to be done in the case where
      // the data is oriented differently. This needs to improved later.
      // Note: avoiding the same-orientation-on-both-sides copy is possible
      // even with AMR since the quantities are already on the mortar at this
      // point
      const auto neighbor_packaged_data = orient_variables_on_slice(
          packaged_data, boundary_extents, dimension, orientation);

      for (const auto& neighbor : neighbors_in_direction) {
        Parallel::receive_data<
            typename ComputeBoundaryFlux<Metavariables>::FluxesTag>(
            receiver_proxy[neighbor], time_id,
            std::make_pair(
                std::make_pair(direction_from_neighbor, element.id()),
                neighbor_packaged_data));

        mortars_local_data.emplace(std::make_pair(direction, neighbor),
                                   local_data);
      }  // loop over neighbors_in_direction
    }    // loop over element.neighbors()

    return std::make_tuple(
        db::create_from<
            db::RemoveTags<interface_normal_dot_fluxes_tag>,
            db::AddSimpleTags<typename ReceiverAction::mortars_local_data_tag>>(
            box, std::move(mortars_local_data)));
  }
};
}  // namespace Actions
}  // namespace dg
