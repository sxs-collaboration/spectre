// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/Direction.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"

namespace DgActions_detail {

template <typename Metavariables, typename DataBoxType, typename DirectionsTag,
          typename NumericalFlux, size_t VolumeDim = DirectionsTag::volume_dim,
          typename FluxCommTypes = dg::FluxCommunicationTypes<Metavariables>,
          typename PackagedData = typename FluxCommTypes::PackagedData>
DirectionMap<VolumeDim, PackagedData> compute_packaged_data(
    const DataBoxType& box,
    const NumericalFlux& normal_dot_numerical_flux_computer,
    const DirectionsTag /*meta*/, const Metavariables /*meta*/) noexcept {
  return interface_apply<
      DirectionsTag,
      tmpl::flatten<tmpl::list<domain::Tags::Mesh<VolumeDim - 1>,
                               typename NumericalFlux::argument_tags>>,
      get_volume_tags<NumericalFlux>>(
      [&normal_dot_numerical_flux_computer](
          const ::Mesh<VolumeDim - 1>& face_mesh,
          const auto&... args) noexcept {
        PackagedData result{face_mesh.number_of_grid_points()};
        normal_dot_numerical_flux_computer.package_data(make_not_null(&result),
                                                        args...);
        return result;
      },
      box);
}

template <typename Metavariables, typename DataBoxType, typename DirectionsTag,
          typename NumericalFlux, size_t VolumeDim = DirectionsTag::volume_dim,
          typename FluxCommTypes = dg::FluxCommunicationTypes<Metavariables>,
          typename LocalData = typename FluxCommTypes::LocalData>
DirectionMap<VolumeDim, LocalData> compute_local_mortar_data(
    const DataBoxType& box,
    const NumericalFlux& normal_dot_numerical_flux_computer,
    const DirectionsTag /*meta*/, const Metavariables /*meta*/) noexcept {
  using normal_dot_fluxes_tag =
      domain::Tags::Interface<DirectionsTag,
                              typename FluxCommTypes::normal_dot_fluxes_tag>;

  const auto& face_meshes =
      db::get<domain::Tags::Interface<DirectionsTag,
                                      domain::Tags::Mesh<VolumeDim - 1>>>(box);
  const auto& magnitude_of_face_normals = db::get<domain::Tags::Interface<
      DirectionsTag,
      Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<VolumeDim>>>>(box);
  const auto& normal_dot_fluxes = db::get<normal_dot_fluxes_tag>(box);

  const auto packaged_data =
      compute_packaged_data(box, normal_dot_numerical_flux_computer,
                            DirectionsTag{}, Metavariables{});

  DirectionMap<VolumeDim, LocalData> all_interface_data{};
  for (const auto& direction : get<DirectionsTag>(box)) {
    LocalData interface_data{};
    interface_data.magnitude_of_face_normal =
        magnitude_of_face_normals.at(direction);
    interface_data.mortar_data.initialize(
        face_meshes.at(direction).number_of_grid_points());
    interface_data.mortar_data.assign_subset(packaged_data.at(direction));

    if (tmpl::size<typename FluxCommTypes::LocalMortarData::tags_list>::value !=
        tmpl::size<typename FluxCommTypes::PackagedData::tags_list>::value) {
      // The local fluxes were not (all) included in the packaged
      // data, so we need to add them to the mortar data
      // explicitly.
      interface_data.mortar_data.assign_subset(normal_dot_fluxes.at(direction));
    }
    all_interface_data.insert({direction, std::move(interface_data)});
  }
  return all_interface_data;
}
}  // namespace DgActions_detail
