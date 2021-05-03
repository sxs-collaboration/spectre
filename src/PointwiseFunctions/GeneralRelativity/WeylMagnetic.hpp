// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

//@(
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the magnetic part of the Weyl tensor.
 *
 * \details Computes the magnetic part of the Weyl tensor \f$B_{ij}\f$
 * as: \f$ B_{ij} =
 \left(1/\sqrt{\det\gamma}\right)D_{k}K_{l(i}\gamma_{j)m}\epsilon^{mlk}
 * \f$ where \f$\epsilon^{ijk}\f$ is the spatial Levi-Civita symbol,
 \f$K_{ij}\f$
 * is the extrinsic curvature, \f$\gamma_{jm} \f$ is the spatial metric,
 * and \f$D_i\$f is spatial covariant derivative.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_magnetic(
    const tnsr::ijj<DataType, SpatialDim, Frame>& grad_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_magnetic(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> weyl_magnetic_part,
    const tnsr::ijj<DataType, SpatialDim, Frame>& grad_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept;
//@}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the scalar \f$B_{ij} B^{ij}\f$ from the magnetic
 * part of the Weyl tensor \f$B_{ij}\f$.
 *
 * \details Computes the scalar \f$B_{ij} B^{ij}\f$ from the magnetic part
 * of the Weyl tensor \f$B_{ij}\f$ and the inverse spatial metric
 * \f$g^{ij}\f$, i.e. \f$B_{ij} = \gamma^{ik}\gamma^{jl}B_{ij}B_{kl}\f$.
 *
 * \note The magnetic part of the Weyl tensor in vacuum is available via
 * gr::weyl_magnetic(). The magnetic part of the Weyl tensor needs additional
 * terms for matter.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weyl_magnetic_scalar(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_magnetic_scalar(
    gsl::not_null<Scalar<DataType>*> weyl_magnetic_scalar_result,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_magenetic,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;
// @}

namespace Tags {
/// Compute item for the magnetic part of the weyl tensor in vacuum
/// Computed from the SpatialRicci, ExtrinsicCurvature, and InverseSpatialMetric
///
/// Can be retrieved using gr::Tags::WeylMagnetic
template <size_t SpatialDim, typename Frame, typename DataType>
struct WeylMagneticCompute : WeylMagnetic<SpatialDim, Frame, DataType>,
                             db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<gr::Tags::ExtrinsicCurvature<SpatialDim, Frame, DataType>,
                    tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::SpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = tnsr::ii<DataType, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>,
      const tnsr::ijj<DataType, SpatialDim, Frame>&,
      const tnsr::ii<DataType, SpatialDim, Frame>&)>(
      &weyl_magnetic<SpatialDim, Frame, DataType>);

  using base = WeylMagnetic<SpatialDim, Frame, DataType>;
};

/// Can be retrieved using gr::Tags::WeylMagneticScalar
template <size_t SpatialDim, typename Frame, typename DataType>
struct WeylMagneticScalarCompute : WeylMagneticScalar<DataType>,
                                   db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::WeylMagneticCompute<SpatialDim, Frame, DataType>,
                 gr::Tags::InverseSpatialMetric<SpatialDim, Frame, DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataType>*>,
                           const tnsr::ii<DataType, SpatialDim, Frame>&,
                           const tnsr::II<DataType, SpatialDim, Frame>&)>(
          &gr::weyl_magnetic_scalar<SpatialDim, Frame, DataType>);

  using base = WeylMagneticScalar<DataType>;
};
}  // namespace Tags
}  // namespace gr
