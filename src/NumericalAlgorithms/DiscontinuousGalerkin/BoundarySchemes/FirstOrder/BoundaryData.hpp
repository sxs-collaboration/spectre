// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace FirstOrderScheme {

namespace detail {
template <typename NumericalFluxType>
struct BoundaryDataImpl {
  static_assert(tt::conforms_to_v<NumericalFluxType, protocols::NumericalFlux>,
                "The 'NumericalFluxType' must conform to the "
                "'dg::protocols::NumericalFlux'.");
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
}  // namespace FirstOrderScheme
}  // namespace dg
