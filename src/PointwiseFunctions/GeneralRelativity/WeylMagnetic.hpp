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
 * \ingroup GeneralRelativityGroup
 * \brief Computes the magnetic part of the Weyl tensor.
 *
 * \details Computes the magnetic part of the Weyl tensor \f$B_{ij}\f$
 * as:
 *
 * \f{align}{
 * B_{ij} =
 * \left(1/\sqrt{\det\gamma}\right)D_{k}K_{l(i}\gamma_{j)m}\epsilon^{mlk} \f}
 *
 * where \f$\epsilon^{ijk}\f$ is the spatial Levi-Civita symbol,
 * \f$K_{ij}\f$
 * is the extrinsic curvature, \f$\gamma_{jm} \f$ is the spatial metric,
 * and \f$D_i\f$ is spatial covariant derivative.
 */
template <typename Frame, typename DataType>
tnsr::ii<DataType, 3, Frame> weyl_magnetic(
    const tnsr::ijj<DataType, 3, Frame>& grad_extrinsic_curvature,
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric);

template <typename Frame, typename DataType>
void weyl_magnetic(
    gsl::not_null<tnsr::ii<DataType, 3, Frame>*> weyl_magnetic_part,
    const tnsr::ijj<DataType, 3, Frame>& grad_extrinsic_curvature,
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the scalar \f$B_{ij} B^{ij}\f$ from the magnetic
 * part of the Weyl tensor \f$B_{ij}\f$.
 *
 * \details Computes the scalar \f$B_{ij} B^{ij}\f$ from the magnetic part
 * of the Weyl tensor \f$B_{ij}\f$ and the inverse spatial metric
 * \f$\gamma^{ij}\f$, i.e. \f$B_{ij} = \gamma^{ik}\gamma^{jl}B_{ij}B_{kl}\f$.
 *
 * \note The magnetic part of the Weyl tensor in vacuum is available via
 * `gr::weyl_magnetic()`. The magnetic part of the Weyl tensor needs additional
 * terms for matter.
 */
template <typename Frame, typename DataType>
Scalar<DataType> weyl_magnetic_scalar(
    const tnsr::ii<DataType, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric);

template <typename Frame, typename DataType>
void weyl_magnetic_scalar(
    gsl::not_null<Scalar<DataType>*> weyl_magnetic_scalar_result,
    const tnsr::ii<DataType, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric);
/// @}

namespace Tags {
/// Compute item for the magnetic part of the weyl tensor in vacuum
/// Computed from the `ExtrinsicCurvature` and `SpatialMetric`
///
/// Can be retrieved using gr::Tags::WeylMagnetic
template <typename DataType, size_t Dim, typename Frame>
struct WeylMagneticCompute : WeylMagnetic<DataType, Dim, Frame>,
                             db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<gr::Tags::ExtrinsicCurvature<DataType, Dim, Frame>,
                    tmpl::size_t<Dim>, Frame>,
      gr::Tags::SpatialMetric<DataType, Dim, Frame>,
      gr::Tags::SqrtDetSpatialMetric<DataType>>;

  using return_type = tnsr::ii<DataType, Dim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataType, Dim, Frame>*>,
      const tnsr::ijj<DataType, Dim, Frame>&,
      const tnsr::ii<DataType, Dim, Frame>&, const Scalar<DataType>&)>(
      &weyl_magnetic<Frame, DataType>);

  using base = WeylMagnetic<DataType, Dim, Frame>;
};

/// Can be retrieved using gr::Tags::`WeylMagneticScalar`
/// Computes magnetic part of the Weyl tensor
template <typename DataType, size_t Dim, typename Frame>
struct WeylMagneticScalarCompute : WeylMagneticScalar<DataType>,
                                   db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylMagneticCompute<DataType, Dim, Frame>,
                 gr::Tags::InverseSpatialMetric<DataType, Dim, Frame>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::ii<DataType, Dim, Frame>&,
      const tnsr::II<DataType, Dim, Frame>&)>(
      &gr::weyl_magnetic_scalar<Frame, DataType>);

  using base = WeylMagneticScalar<DataType>;
};
}  // namespace Tags
}  // namespace gr
