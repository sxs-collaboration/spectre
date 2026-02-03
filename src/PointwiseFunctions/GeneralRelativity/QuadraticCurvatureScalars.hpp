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
 * \brief Computes the Pontryagin scalar in vacuum.
 *
 * \details The Pontryagin scalar in vacuum is given by
 * \begin{align}
 *   \mathcal{P} &\equiv {^{\star} C}_{abcd} C^{abcd} \\
 *    &= - 16 E_{ab} B^{ab} ~,
 * \end{align}
 * where $ C_{abcd} $ it the Weyl tensor (with dual $ {^\star} C}_{abcd} $) in 4
 * spacetime dimensions. Here it is computed in terms of the electric ($
 * E_{ab} $) and magnetic ($ B_{ab} $) parts of the Weyl scalar, with the
 * conventions used here for ($ \{E_{ab}, B_{ab}\} $).
 *
 * \see `gr::Tags::WeylMagnetic` and `gr::Tags::WeylElectric`
 *
 */
template <typename Frame>
void pontryagin_scalar_in_vacuum(
    gsl::not_null<Scalar<DataVector>*> pontryagin_scalar,
    const tnsr::ii<DataVector, 3, Frame>& weyl_electric,
    const tnsr::ii<DataVector, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric);

template <typename Frame>
Scalar<DataVector> pontryagin_scalar_in_vacuum(
    const tnsr::ii<DataVector, 3, Frame>& weyl_electric,
    const tnsr::ii<DataVector, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric);
/// @}

/// @{
/*!
 * \brief Computes Gauss-Bonnet scalar in vacuum.
 *
 * \details The Gauss-Bonnet scalar in vacuum is given by
 * \begin{align}
 *   \mathcal{G} &\equiv R_{abcd} R^{abcd} - 4 R_{ab} R^{ab} + R^2
 *    &= C_{abcd} C^{abcd}
 *    &= 8 (E_{ab} E^{ab} - B_{ab} B^{ab}) ~,
 * \end{align}
 * where $ R_{abcd} $, $ R_{ab} $, $ R $ $ C_{abcd} $ are the Riemann tensor,
 * Ricci tensor, Ricci scalar and Weyl tensor in 4 spacetime dimensions. The
 * Gauss-Bonnet scalar in vacuum can be computed in terms of the electric ($
 * E_{ab} $) and magnetic ($ B_{ab} $) parts of the Weyl tensor.
 *
 * \see `gr::Tags::WeylMagnetic` and `gr::Tags::WeylElectric`
 *
 */
void gauss_bonnet_scalar_in_vacuum(
    gsl::not_null<Scalar<DataVector>*> gb_scalar,
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar);

Scalar<DataVector> gauss_bonnet_scalar_in_vacuum(
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar);
/// @}

}  // namespace gr

namespace gr::Tags {
/// @{
/*!
 * \brief Tags for the PontryaginScalar in vacuum.
 *
 * The tags are tested in Test_CurvatureScalarComputeTags.cpp
 */
template <typename DataType>
struct PontryaginScalar : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType, size_t Dim, typename Frame>
struct PontryaginScalarCompute : PontryaginScalar<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylElectric<DataType, Dim, Frame>,
                 gr::Tags::WeylMagnetic<DataType, Dim, Frame>,
                 gr::Tags::InverseSpatialMetric<DataType, Dim, Frame>>;
  using return_type = Scalar<DataType>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::ii<DataType, Dim, Frame>&,
      const tnsr::ii<DataType, Dim, Frame>&,
      const tnsr::II<DataType, Dim, Frame>&)>(
      &gr::pontryagin_scalar_in_vacuum<Frame>);
  using base = PontryaginScalar<DataType>;
};
/// @{

/// @{
/*!
 * \brief Tags for the Gauss Bonnet Scalar in vacuum.
 *
 * The tags are tested in Test_CurvatureScalarComputeTags.cpp
 */
template <typename DataType>
struct GaussBonnetScalar : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct GaussBonnetScalarCompute : GaussBonnetScalar<DataType>, db::ComputeTag {
  using argument_tags = tmpl::list<gr::Tags::WeylElectricScalar<DataType>,
                                   gr::Tags::WeylMagneticScalar<DataType>>;
  using return_type = Scalar<DataType>;
  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataType>*>,
                           const Scalar<DataType>&, const Scalar<DataType>&)>(
          &gr::gauss_bonnet_scalar_in_vacuum);
  using base = GaussBonnetScalar<DataType>;
};
/// @}
}  // namespace gr::Tags
