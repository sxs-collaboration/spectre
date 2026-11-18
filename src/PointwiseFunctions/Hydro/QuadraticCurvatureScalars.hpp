// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/QuadraticCurvatureScalars.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
 * \brief Computes Kretschmann scalar in hydro.
 *
 * \details The Kretschmann scalar is given by
 * \f{align}{
 *   \mathcal{K} &\equiv R_{abcd} R^{abcd} \\
 *    &= C_{abcd} C^{abcd} + 2R_{ab}R^{ab} - R^2/3 \\
 *    &= 8 (E_{ab} E^{ab} - B_{ab} B^{ab}) + 2R_{ab}R^{ab} - R^2/3 ~,
 * \f}
 * where \f$R_{abcd}\f$, \f$C_{abcd}\f$ are the Riemann tensor and Weyl tensor
 * in 4 spacetime dimensions, \f$R_{ab}\f$ is the Ricci tensor, \f$R\f$ is the
 * Ricci scalar, and \f$E_{ab}\f$, \f$B_{ab}\f$ are the electric and magnetic
 * parts of the Weyl tensor.
 *
 * \see `gr::Tags::WeylMagnetic` and `gr::Tags::WeylElectric`
 *
 */
template <typename DataType>
void kretschmann_scalar(gsl::not_null<Scalar<DataType>*> result,
                        const Scalar<DataType>& weyl_electric_scalar,
                        const Scalar<DataType>& weyl_magnetic_scalar,
                        const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
                        const tnsr::aa<DataType, 3>& ricci_tensor,
                        const Scalar<DataType>& ricci_scalar);

template <typename DataType>
Scalar<DataType> kretschmann_scalar(
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar,
    const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
    const tnsr::aa<DataType, 3>& ricci_tensor,
    const Scalar<DataType>& ricci_scalar);
/// @}

namespace Tags {
/// Compute item for the Kretschmann scalar in hydro.
///
/// Can be retrieved using hydro::Tags::KretschmannScalar
template <typename DataType>
struct KretschmannScalarCompute : KretschmannScalar<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<hydro::Tags::WeylElectricScalar<DataType>,
                 gr::Tags::WeylMagneticScalar<DataType>,
                 gr::Tags::InverseSpacetimeMetric<DataType, 3>,
                 hydro::Tags::GrRicci<DataType, 3>,
                 hydro::Tags::GrRicciScalar<DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const Scalar<DataType>&,
      const Scalar<DataType>&, const tnsr::AA<DataType, 3>&,
      const tnsr::aa<DataType, 3>&, const Scalar<DataType>&)>(
      &hydro::kretschmann_scalar);

  using base = KretschmannScalar<DataType>;
};

/// Compute item for the Pontryagin scalar in hydro.
///
/// Can be retrieved using hydro::Tags::PontryaginScalar
template <typename DataType>
struct PontryaginScalarCompute : PontryaginScalar<DataType>, db::ComputeTag {
  using argument_tags = tmpl::list<hydro::Tags::WeylElectric<DataType, 3>,
                                   gr::Tags::WeylMagnetic<DataType, 3>,
                                   gr::Tags::InverseSpatialMetric<DataType, 3>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::ii<DataType, 3>&,
      const tnsr::ii<DataType, 3>&, const tnsr::II<DataType, 3>&)>(
      &gr::pontryagin_scalar);

  using base = PontryaginScalar<DataType>;
};
}  // namespace Tags
}  // namespace hydro
