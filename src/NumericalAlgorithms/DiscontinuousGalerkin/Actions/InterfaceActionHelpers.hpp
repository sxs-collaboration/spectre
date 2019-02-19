// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/Direction.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"

namespace DgActions_detail {

template <typename NumericalFlux, typename Metavariables, typename... CacheTags,
          typename... Args>
void compute_packaged_data_with_cache_tags(
    const gsl::not_null<Variables<typename NumericalFlux::package_tags>*>
        packaged_data,
    const NumericalFlux& normal_dot_numerical_flux_computer,
    const Parallel::ConstGlobalCache<Metavariables>& cache,
    tmpl::list<CacheTags...> /*meta*/, const Args&... args) {
  normal_dot_numerical_flux_computer.package_data(packaged_data, args...,
                                                  get<CacheTags>(cache)...);
}

template <typename Metavariables, typename DataBoxType, typename DirectionsTag,
          typename NumericalFlux>
auto compute_packaged_data(
    const DataBoxType& box,
    const Direction<Metavariables::system::volume_dim>& direction,
    const NumericalFlux& normal_dot_numerical_flux_computer,
    const DirectionsTag /*meta*/,
    const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept ->
    typename dg::FluxCommunicationTypes<Metavariables>::PackagedData {
  constexpr size_t volume_dim = Metavariables::system::volume_dim;

  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;

  using package_arguments =
      typename Metavariables::normal_dot_numerical_flux::type::argument_tags;

  const auto& face_mesh =
      db::get<Tags::Interface<DirectionsTag, Tags::Mesh<volume_dim - 1>>>(box)
          .at(direction);

  return db::apply<tmpl::transform<
      package_arguments, tmpl::bind<Tags::Interface, DirectionsTag, tmpl::_1>>>(
      [&face_mesh, &direction, &normal_dot_numerical_flux_computer,
       &cache ](const auto&... args) noexcept {
        typename flux_comm_types::PackagedData ret(
            face_mesh.number_of_grid_points(), 0.0);
        compute_packaged_data_with_cache_tags(
            make_not_null(&ret), normal_dot_numerical_flux_computer, cache,
            typename NumericalFlux::const_global_cache_tags{},
            args.at(direction)...);
        return ret;
      },
      box);
}

template <typename Metavariables, typename DataBoxType, typename DirectionsTag,
          typename NumericalFlux>
auto compute_local_mortar_data(
    const DataBoxType& box,
    const Direction<Metavariables::system::volume_dim>& direction,
    const NumericalFlux& normal_dot_numerical_flux_computer,
    const DirectionsTag /*meta*/,
    const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept ->
    typename dg::FluxCommunicationTypes<Metavariables>::LocalData {
  constexpr size_t volume_dim = Metavariables::system::volume_dim;

  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;

  using normal_dot_fluxes_tag =
      Tags::Interface<DirectionsTag,
                      typename flux_comm_types::normal_dot_fluxes_tag>;

  const auto& face_mesh =
      db::get<Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
                              Tags::Mesh<volume_dim - 1>>>(box)
          .at(direction);

  const auto packaged_data =
      compute_packaged_data(box, direction, normal_dot_numerical_flux_computer,
                            DirectionsTag{}, cache);

  typename flux_comm_types::LocalData interface_data{};
  interface_data.magnitude_of_face_normal = db::get<Tags::Interface<
      DirectionsTag,
      Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>>(box)
                                                .at(direction);

  interface_data.mortar_data.initialize(face_mesh.number_of_grid_points());
  interface_data.mortar_data.assign_subset(packaged_data);

  if (tmpl::size<typename flux_comm_types::LocalMortarData::tags_list>::value !=
      tmpl::size<typename flux_comm_types::PackagedData::tags_list>::value) {
    // The local fluxes were not (all) included in the packaged
    // data, so we need to add them to the mortar data
    // explicitly.
    const auto& normal_dot_fluxes =
        db::get<normal_dot_fluxes_tag>(box).at(direction);
    interface_data.mortar_data.assign_subset(normal_dot_fluxes);
  }

  return interface_data;
}
}  // namespace DgActions_detail
