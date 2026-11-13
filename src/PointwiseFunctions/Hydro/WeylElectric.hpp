// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace hydro {

/// @{
/*!
 * \brief Computes the electric part of the Weyl tensor in a hydro system.
 *
 * \details Computes the electric part of the Weyl tensor \f$E_{ij}\f$ in hydro
 * as the spatial components of
 *
 * \f{align}{
 *   E_{ab} = E_{ab}^{(\mathrm{vac})} -
 *   \frac{1}{2}(\gamma_a^{\ c}\gamma_b^{\ d} +
 *   \gamma_{ab}\gamma^{cd}){}^{(4)}R_{cd} +
 *   \frac{1}{3}\gamma_{ab}{}^{(4)}R ~,
 * \f}
 *
 * where \f$E_{ab}^{(\mathrm{vac})}\f$ is the vacuum contribution to the
 * electric part of the Weyl tensor (computed in `GeneralRelativity`),
 * \f$\gamma_{ab}\f$ is the induced spatial metric, \f${}^{(4)}R_{ab}\f$ is
 * the Ricci tensor, and \f${}^{(4)}R\f$ is the Ricci scalar.
 */
template <typename DataType>
tnsr::ii<DataType, 3> weyl_electric(
    const tnsr::ii<DataType, 3>& vacuum_weyl_electric,
    const tnsr::AA<DataType, 3>& stress_energy,
    const tnsr::aa<DataType, 3>& ricci_tensor,
    const Scalar<DataType>& ricci_scalar,
    const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
    const tnsr::aa<DataType, 3>& induced_spatial_metric);

template <typename DataType>
void weyl_electric(gsl::not_null<tnsr::ii<DataType, 3>*> weyl_electric,
                   const tnsr::ii<DataType, 3>& vacuum_weyl_electric,
                   const tnsr::AA<DataType, 3>& stress_energy,
                   const tnsr::aa<DataType, 3>& ricci_tensor,
                   const Scalar<DataType>& ricci_scalar,
                   const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
                   const tnsr::aa<DataType, 3>& induced_spatial_metric);
/// @}

namespace Tags {
/// Compute item for the electric part of the weyl tensor in hydro.
///
/// Can be retrieved using hydro::Tags::WeylElectric
template <typename DataType>
struct WeylElectricCompute : WeylElectric<DataType, 3>, db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylElectric<DataType, 3>,
                 hydro::Tags::StressEnergy<DataType, 3>,
                 hydro::Tags::GrRicci<DataType, 3>,
                 hydro::Tags::GrRicciScalar<DataType>,
                 gr::Tags::InverseSpacetimeMetric<DataType, 3>,
                 gr::Tags::InducedSpatialMetric<DataType, 3>>;

  using return_type = tnsr::ii<DataType, 3>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataType, 3>*>, const tnsr::ii<DataType, 3>&,
      const tnsr::AA<DataType, 3>&, const tnsr::aa<DataType, 3>&,
      const Scalar<DataType>&, const tnsr::AA<DataType, 3>&,
      const tnsr::aa<DataType, 3>&)>(&weyl_electric);

  using base = WeylElectric<DataType, 3>;
};

/// Compute item for the Weyl electric scalar in hydro.
///
/// Can be retrieved using hydro::Tags::WeylElectricScalar
template <typename DataType>
struct WeylElectricScalarCompute : WeylElectricScalar<DataType>,
                                   db::ComputeTag {
  using argument_tags = tmpl::list<hydro::Tags::WeylElectric<DataType, 3>,
                                   gr::Tags::InverseSpatialMetric<DataType, 3>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::ii<DataType, 3>&,
      const tnsr::II<DataType, 3>&)>(&gr::weyl_electric_scalar<DataType, 3>);

  using base = WeylElectricScalar<DataType>;
};
}  // namespace Tags
}  // namespace hydro
