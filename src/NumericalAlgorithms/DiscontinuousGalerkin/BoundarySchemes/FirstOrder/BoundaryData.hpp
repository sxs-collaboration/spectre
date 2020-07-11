// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace FirstOrderScheme {

namespace detail {
template <typename NumericalFluxType>
struct BoundaryDataImpl {
  static_assert(
      tt::assert_conforms_to<NumericalFluxType, protocols::NumericalFlux>);
  using type = dg::SimpleBoundaryData<
      tmpl::remove_duplicates<tmpl::append<
          db::wrap_tags_in<::Tags::NormalDotFlux,
                           typename NumericalFluxType::variables_tags>,
          typename NumericalFluxType::package_field_tags>>,
      typename NumericalFluxType::package_extra_tags>;
};
}  // namespace detail

/*!
 * \brief The data on element boundaries that's needed for the (strong)
 * first-order boundary scheme
 *
 * The boundary data includes the `NumericalFluxType`'s packaged data plus all
 * "normal-dot-fluxes" so the strong boundary scheme can compute the difference
 * between the normal-dot-numerical-fluxes and the normal-dot-fluxes
 * (see `dg::FirstOrder`). For the weak formulation the normal-dot-fluxes
 * need not be included explicitly, but the `NumericalFluxType` may require
 * a subset of them.
 */
template <typename NumericalFluxType>
using BoundaryData = typename detail::BoundaryDataImpl<NumericalFluxType>::type;

/*!
 * \brief Package the data on element boundaries that's needed for the (strong)
 * first-order boundary scheme.
 *
 * This function currently packages the data required by the `NumericalFluxType`
 * plus all "normal-dot-fluxes", which are needed for the strong first-order
 * boundary scheme (see `dg::FirstOrderScheme::BoundaryData`). Note that for
 * the weak formulation the normal-dot-fluxes need not be included explicitly,
 * but the `NumericalFluxType` may require a subset of them.
 */
template <
    size_t FaceDim, typename NumericalFluxType, typename... NumericalFluxArgs,
    Requires<tt::conforms_to_v<NumericalFluxType, protocols::NumericalFlux>> =
        nullptr>
auto package_boundary_data(
    const NumericalFluxType& numerical_flux_computer,
    const Mesh<FaceDim>& face_mesh,
    const Variables<db::wrap_tags_in<
        ::Tags::NormalDotFlux, typename NumericalFluxType::variables_tags>>&
        normal_dot_fluxes,
    const NumericalFluxArgs&... args) noexcept {
  BoundaryData<NumericalFluxType> boundary_data{
      face_mesh.number_of_grid_points()};
  boundary_data.field_data.assign_subset(normal_dot_fluxes);
  dg::NumericalFluxes::package_data(make_not_null(&boundary_data),
                                    numerical_flux_computer, args...);
  return boundary_data;
}

}  // namespace FirstOrderScheme
}  // namespace dg
