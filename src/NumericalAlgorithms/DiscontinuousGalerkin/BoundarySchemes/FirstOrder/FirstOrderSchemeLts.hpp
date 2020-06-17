// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace FirstOrderScheme {

namespace detail {

template <size_t Dim, typename VariablesTag, typename NumericalFluxComputerTag,
          typename BoundaryData,
          typename NumericalFlux = typename NumericalFluxComputerTag::type,
          typename ArgsTagsList = typename NumericalFlux::argument_tags>
struct boundary_data_computer_lts_impl;

template <size_t Dim, typename VariablesTag, typename NumericalFluxComputerTag,
          typename BoundaryData, typename NumericalFlux, typename... ArgsTags>
struct boundary_data_computer_lts_impl<Dim, VariablesTag,
                                       NumericalFluxComputerTag, BoundaryData,
                                       NumericalFlux, tmpl::list<ArgsTags...>> {
  using n_dot_fluxes_tag =
      db::add_tag_prefix<::Tags::NormalDotFlux, VariablesTag>;
  using magnitude_of_face_normal_tag =
      ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>;
  using argument_tags = tmpl::list<NumericalFluxComputerTag, n_dot_fluxes_tag,
                                   magnitude_of_face_normal_tag, ArgsTags...>;
  using volume_tags = tmpl::append<tmpl::list<NumericalFluxComputerTag>,
                                   get_volume_tags<NumericalFlux>>;
  static auto apply(const NumericalFlux& numerical_flux_computer,
                    const typename n_dot_fluxes_tag::type& normal_dot_fluxes,
                    const Scalar<DataVector>& face_normal_magnitude,
                    const typename ArgsTags::type&... args) noexcept {
    BoundaryData boundary_data{normal_dot_fluxes.number_of_grid_points()};
    boundary_data.field_data.assign_subset(normal_dot_fluxes);
    dg::NumericalFluxes::package_data(make_not_null(&boundary_data),
                                      numerical_flux_computer, args...);
    get<magnitude_of_face_normal_tag>(boundary_data.extra_data) =
        face_normal_magnitude;
    return boundary_data;
  }
};

}  // namespace detail

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Boundary contributions for a first-order DG scheme with local
 * time-stepping
 *
 * This class is the local time-stepping equivalent to the
 * `dg::FirstOrderScheme::FirstOrderScheme`. Notable differences are:
 *
 * - Boundary contributions are added to the `VariablesTag` directly.
 * - We use the `Tags::BoundaryHistory` on mortars.
 * - We need to store the face-normal magnitude in the boundary history, as
 * opposed to the non-LTS case where we can retrieve it when needed.
 */
template <size_t Dim, typename VariablesTag, typename NumericalFluxComputerTag,
          typename TemporalIdTag, typename TimeStepperTag>
struct FirstOrderSchemeLts {
 private:
  using base = FirstOrderScheme<Dim, VariablesTag, NumericalFluxComputerTag,
                                TemporalIdTag>;

 public:
  static constexpr size_t volume_dim = Dim;
  using variables_tag = VariablesTag;
  using numerical_flux_computer_tag = NumericalFluxComputerTag;
  using NumericalFlux = typename NumericalFluxComputerTag::type;
  using temporal_id_tag = TemporalIdTag;
  using receive_temporal_id_tag = ::Tags::Next<temporal_id_tag>;
  using time_stepper_tag = TimeStepperTag;

  using magnitude_of_face_normal_tag =
      ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<volume_dim>>;
  using BoundaryData = dg::SimpleBoundaryData<
      typename base::BoundaryData::field_tags,
      tmpl::push_back<typename base::BoundaryData::extra_data_tags,
                      magnitude_of_face_normal_tag>>;
  using boundary_data_computer = detail::boundary_data_computer_lts_impl<
      volume_dim, variables_tag, numerical_flux_computer_tag, BoundaryData>;

  using mortar_data_tag = Tags::BoundaryHistory<BoundaryData, BoundaryData,
                                                typename variables_tag::type>;

  using return_tags =
      tmpl::list<variables_tag, ::Tags::Mortars<mortar_data_tag, Dim>>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>,
                 ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
                 NumericalFluxComputerTag, time_stepper_tag, ::Tags::TimeStep>;

  static void apply(
      const gsl::not_null<db::item_type<variables_tag>*> variables,
      const gsl::not_null<db::item_type<::Tags::Mortars<mortar_data_tag, Dim>>*>
          all_mortar_data,
      const Mesh<Dim>& volume_mesh,
      const typename ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>::type&
          mortar_meshes,
      const typename ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>::type&
          mortar_sizes,
      const NumericalFlux& normal_dot_numerical_flux_computer,
      const typename time_stepper_tag::type::element_type& time_stepper,
      const TimeDelta& time_step) noexcept {
    // Iterate over all mortars
    for (auto& mortar_id_and_data : *all_mortar_data) {
      // Retrieve mortar data
      const auto& mortar_id = mortar_id_and_data.first;
      auto& mortar_data = mortar_id_and_data.second;
      const auto& direction = mortar_id.first;
      const size_t dimension = direction.dimension();

      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const auto& mortar_size = mortar_sizes.at(mortar_id);
      const auto face_mesh = volume_mesh.slice_away(dimension);
      const size_t extent_perpendicular_to_face =
          volume_mesh.extents(dimension);

      // This lambda must only capture quantities that are
      // independent of the simulation state.
      const auto coupling = [&face_mesh, &mortar_mesh, &mortar_size,
                             &extent_perpendicular_to_face,
                             &normal_dot_numerical_flux_computer](
                                const BoundaryData& local_data,
                                const BoundaryData& remote_data) noexcept {
        return boundary_flux(
            local_data, remote_data, normal_dot_numerical_flux_computer,
            get<magnitude_of_face_normal_tag>(local_data.extra_data),
            extent_perpendicular_to_face, face_mesh, mortar_mesh, mortar_size);
      };

      const auto lifted_data = time_stepper.compute_boundary_delta(
          coupling, make_not_null(&mortar_data), time_step);

      // Add the flux contribution to the volume data
      add_slice_to_data(variables, lifted_data, volume_mesh.extents(),
                        dimension,
                        index_to_slice_at(volume_mesh.extents(), direction));
    }
  }
};

}  // namespace FirstOrderScheme
}  // namespace dg
