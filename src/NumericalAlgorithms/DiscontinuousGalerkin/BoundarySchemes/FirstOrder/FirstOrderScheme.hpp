// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace FirstOrderScheme {

namespace detail {

template <size_t Dim, typename VariablesTag, typename NumericalFluxComputerTag>
struct boundary_data_computer_impl {
  using NumericalFlux = typename NumericalFluxComputerTag::type;
  using n_dot_fluxes_tag =
      db::add_tag_prefix<::Tags::NormalDotFlux, VariablesTag>;
  using argument_tags =
      tmpl::push_front<typename NumericalFlux::argument_tags,
                       NumericalFluxComputerTag, domain::Tags::Mesh<Dim - 1>,
                       n_dot_fluxes_tag>;
  using volume_tags = tmpl::push_front<get_volume_tags<NumericalFlux>,
                                       NumericalFluxComputerTag>;
  template <typename... Args>
  static auto apply(const NumericalFlux& numerical_flux_computer,
                    const Mesh<Dim - 1>& face_mesh,
                    const typename n_dot_fluxes_tag::type& normal_dot_fluxes,
                    const Args&... args) noexcept {
    return dg::FirstOrderScheme::package_boundary_data(
        numerical_flux_computer, face_mesh, normal_dot_fluxes, args...);
  }
};

}  // namespace detail

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Boundary contributions for a first-order DG scheme
 *
 * Computes Eq. (2.20) in \cite Teukolsky2015ega and lifts it to the
 * volume (see `dg::lift_flux`) on all mortars that touch an element. The
 * resulting boundary contributions are added to the DG operator data in the
 * `DtVariablesTag`.
 */
template <size_t Dim, typename VariablesTag, typename DtVariablesTag,
          typename NumericalFluxComputerTag, typename TemporalIdTag>
struct FirstOrderScheme {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = VariablesTag;
  using numerical_flux_computer_tag = NumericalFluxComputerTag;
  using NumericalFlux = typename NumericalFluxComputerTag::type;
  using temporal_id_tag = TemporalIdTag;
  using receive_temporal_id_tag = temporal_id_tag;
  using dt_variables_tag = DtVariablesTag;

  static_assert(
      tt::assert_conforms_to<NumericalFlux, dg::protocols::NumericalFlux>);
  // We need the `VariablesTag` as an explicit template parameter only because
  // it may be prefixed.
  static_assert(std::is_same_v<typename NumericalFlux::variables_tags,
                               typename variables_tag::tags_list>,
                "The 'VariablesTag' and the 'NumericalFluxComputerTag' must "
                "have the same list of variables.");

  using BoundaryData = dg::FirstOrderScheme::BoundaryData<NumericalFlux>;
  using boundary_data_computer =
      detail::boundary_data_computer_impl<Dim, VariablesTag,
                                          NumericalFluxComputerTag>;

  using mortar_data_tag = Tags::SimpleMortarData<typename TemporalIdTag::type,
                                                 BoundaryData, BoundaryData>;

  // Only a shortcut
  using magnitude_of_face_normal_tag =
      ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<volume_dim>>;

  using return_tags =
      tmpl::list<dt_variables_tag, ::Tags::Mortars<mortar_data_tag, Dim>>;
  using argument_tags = tmpl::list<
      domain::Tags::Mesh<Dim>,
      ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
      ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
      NumericalFluxComputerTag,
      domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                              magnitude_of_face_normal_tag>,
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                              magnitude_of_face_normal_tag>>;

  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> dt_variables,
      const gsl::not_null<typename ::Tags::Mortars<mortar_data_tag, Dim>::type*>
          all_mortar_data,
      const Mesh<Dim>& volume_mesh,
      const MortarMap<Dim, Mesh<Dim - 1>>& mortar_meshes,
      const MortarMap<Dim, MortarSize<Dim - 1>>& mortar_sizes,
      const NumericalFlux& normal_dot_numerical_flux_computer,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          face_normal_magnitudes_internal,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          face_normal_magnitudes_boundary) noexcept {
    // Iterate over all mortars
    for (auto& mortar_id_and_data : *all_mortar_data) {
      // Retrieve mortar data
      const auto& mortar_id = mortar_id_and_data.first;
      auto& mortar_data = mortar_id_and_data.second;
      const auto& direction = mortar_id.first;
      const size_t dimension = direction.dimension();

      // Extract local and remote data
      const auto extracted_mortar_data = mortar_data.extract();
      const auto& local_data = extracted_mortar_data.first;
      const auto& remote_data = extracted_mortar_data.second;

      const auto& magnitude_of_face_normal =
          mortar_id.second == ElementId<Dim>::external_boundary_id()
              ? face_normal_magnitudes_boundary.at(direction)
              : face_normal_magnitudes_internal.at(direction);

      auto boundary_flux_on_slice =
          typename dt_variables_tag::type(boundary_flux(
              local_data, remote_data, normal_dot_numerical_flux_computer,
              magnitude_of_face_normal, volume_mesh.extents(dimension),
              volume_mesh.slice_away(dimension), mortar_meshes.at(mortar_id),
              mortar_sizes.at(mortar_id)));

      // Add the flux contribution to the volume data
      add_slice_to_data(dt_variables, std::move(boundary_flux_on_slice),
                        volume_mesh.extents(), dimension,
                        index_to_slice_at(volume_mesh.extents(), direction));
    }
  }
};

}  // namespace FirstOrderScheme
}  // namespace dg
