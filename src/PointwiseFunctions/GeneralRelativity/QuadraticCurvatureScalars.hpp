// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

/// @{
/*!
 * \brief Computes the Pontryagin scalar.
 *
 * \details The Pontryagin scalar is given by
 * \f{align}{
 *   \mathcal{P} &\equiv {}^{\star}R_{abcd} R^{abcd} \\
 *    &= {}^{\star}C_{abcd} C^{abcd} \\
 *    &= 16 E_{ab} B^{ab} ~,
 * \f}
 * where \f$C_{abcd}\f$ is the Weyl tensor (with dual \f${}^\star C_{abcd}\f$).
 * Here it is computed in terms of the electric (\f$E_{ab}\f$) and magnetic
 * (\f$B_{ab}\f$) parts of the Weyl tensor, with the
 * conventions used here for (\f$\{E_{ab}, B_{ab}\}\f$).
 * Note that the Pontryagin scalar is insensitive to the Ricci tensor, therefore
 * is not affected by the presence of matter.
 *
 * \see `gr::Tags::WeylMagnetic` and `gr::Tags::WeylElectric`
 *
 * \note The spatial dimension is fixed to 3 to be consistent with
 * `gr::Tags::WeylMagnetic`, which requires the 3D Levi-Civita symbol.
 */
template <typename DataType, typename Frame>
void pontryagin_scalar(
    gsl::not_null<Scalar<DataType>*> pontryagin_scalar,
    const tnsr::ii<DataType, 3, Frame>& weyl_electric,
    const tnsr::ii<DataType, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric);

template <typename DataType, typename Frame>
Scalar<DataType> pontryagin_scalar(
    const tnsr::ii<DataType, 3, Frame>& weyl_electric,
    const tnsr::ii<DataType, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric);
/// @}

/// @{
/*!
 * \brief Computes Kretschmann scalar in vacuum.
 *
 * \details The Kretschmann scalar in vacuum is given by
 * \f{align}{
 *   \mathcal{K} &\equiv R_{abcd} R^{abcd} \\
 *    &= C_{abcd} C^{abcd} \\
 *    &= 8 (E_{ab} E^{ab} - B_{ab} B^{ab}) ~,
 * \f}
 * where \f$R_{abcd}\f$, \f$C_{abcd}\f$ are the Riemann tensor and Weyl tensor
 * in 4 spacetime dimensions. The Kretschmann scalar in vacuum can be computed
 * in terms of the electric (\f$E_{ab}\f$) and magnetic (\f$B_{ab}\f$) parts of
 * the Weyl tensor.
 *
 * \see `gr::Tags::WeylMagnetic` and `gr::Tags::WeylElectric`
 *
 */
template <typename DataType>
void kretschmann_scalar_in_vacuum(
    gsl::not_null<Scalar<DataType>*> kretschmann_scalar,
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar);

template <typename DataType>
Scalar<DataType> kretschmann_scalar_in_vacuum(
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar);
/// @}

/// @{
/*!
 * \brief Computes Gauss-Bonnet scalar in vacuum.
 *
 * \details The Gauss-Bonnet scalar in vacuum is given by
 * \f{align}{
 *   \mathcal{G} &\equiv R_{abcd} R^{abcd} - 4 R_{ab} R^{ab} + R^2 \\
 *    &= C_{abcd} C^{abcd} \\
 *    &= 8 (E_{ab} E^{ab} - B_{ab} B^{ab}) ~,
 * \f}
 * where \f$R_{abcd}\f$, \f$R_{ab}\f$, \f$R\f$, \f$C_{abcd}\f$ are the Riemann
 * tensor, Ricci tensor, Ricci scalar, and Weyl tensor in 4 spacetime
 * dimensions. The Gauss-Bonnet scalar in vacuum can be computed in terms of
 * the electric (\f$E_{ab}\f$) and magnetic (\f$B_{ab}\f$) parts of the Weyl
 * tensor.
 *
 * \see `gr::Tags::WeylMagnetic` and `gr::Tags::WeylElectric`
 *
 */
template <typename DataType>
void gauss_bonnet_scalar_in_vacuum(
    gsl::not_null<Scalar<DataType>*> gb_scalar,
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar);

template <typename DataType>
Scalar<DataType> gauss_bonnet_scalar_in_vacuum(
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar);
/// @}

}  // namespace gr

namespace gr::Tags {
/// @{
/*!
 * \brief Compute tag for the PontryaginScalar in vacuum.
 *
 * The tags are tested in Test_CurvatureScalarComputeTags.cpp
 */

template <typename DataType, typename Frame>
struct PontryaginScalarCompute : PontryaginScalar<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylElectric<DataType, 3, Frame>,
                 gr::Tags::WeylMagnetic<DataType, 3, Frame>,
                 gr::Tags::InverseSpatialMetric<DataType, 3, Frame>>;
  using return_type = Scalar<DataType>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::ii<DataType, 3, Frame>&,
      const tnsr::ii<DataType, 3, Frame>&,
      const tnsr::II<DataType, 3, Frame>&)>(
      &gr::pontryagin_scalar<DataType, Frame>);
  using base = PontryaginScalar<DataType>;
};
/// @}

/// @{
/*!
 * \brief Compute tag for the Kretschmann Scalar in vacuum.
 *
 * The tags are tested in Test_CurvatureScalarComputeTags.cpp
 */

template <typename DataType>
struct KretschmannScalarCompute : KretschmannScalar<DataType>, db::ComputeTag {
  using argument_tags = tmpl::list<gr::Tags::WeylElectricScalar<DataType>,
                                   gr::Tags::WeylMagneticScalar<DataType>>;
  using return_type = Scalar<DataType>;
  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataType>*>,
                           const Scalar<DataType>&, const Scalar<DataType>&)>(
          &gr::kretschmann_scalar_in_vacuum<DataType>);
  using base = KretschmannScalar<DataType>;
};
/// @}

/// @{
/*!
 * \brief Compute tag for the Gauss Bonnet Scalar in vacuum.
 *
 * The tags are tested in Test_CurvatureScalarComputeTags.cpp
 */

template <typename DataType>
struct GaussBonnetScalarCompute : GaussBonnetScalar<DataType>, db::ComputeTag {
  using argument_tags = tmpl::list<gr::Tags::WeylElectricScalar<DataType>,
                                   gr::Tags::WeylMagneticScalar<DataType>>;
  using return_type = Scalar<DataType>;
  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataType>*>,
                           const Scalar<DataType>&, const Scalar<DataType>&)>(
          &gr::gauss_bonnet_scalar_in_vacuum<DataType>);
  using base = GaussBonnetScalar<DataType>;
};
/// @}
}  // namespace gr::Tags
