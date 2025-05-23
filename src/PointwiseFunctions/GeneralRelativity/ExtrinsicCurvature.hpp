// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes extrinsic curvature from metric and derivatives.
 * \details Uses the ADM evolution equation for the spatial metric,
 * \f[ K_{ij} = \frac{1}{2 \alpha} \left ( -\partial_0 \gamma_{ij}
 * + \beta^k \partial_k \gamma_{ij} + \gamma_{ki} \partial_j \beta^k
 * + \gamma_{kj} \partial_i \beta^k \right ) \f]
 * where \f$K_{ij}\f$ is the extrinsic curvature, \f$\alpha\f$ is the lapse,
 * \f$\beta^i\f$ is the shift, and \f$\gamma_{ij}\f$ is the spatial metric. In
 * terms of the Lie derivative of the spatial metric with respect to a unit
 * timelike vector \f$n^a\f$ normal to the spatial slice, this corresponds to
 * the sign convention
 * \f[ K_{ab} = - \frac{1}{2} \mathcal{L}_{\mathbf{n}} \gamma_{ab} \f]
 * where \f$\gamma_{ab}\f$ is the spatial metric. See Eq. (2.53) in
 * \cite BaumgarteShapiro.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
void extrinsic_curvature(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> ex_curvature,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes the spatial covariant derivative of the extrinsic curvature.
 *  \details The spatial covariant derivative is computed as
 * \f[ D_k K_{ij} = \partial_k K_{ij} - {^{(3)}\Gamma^{l}_{ki}} K_{lj}
 * - {^{(3)}\Gamma^{l}_{kj}}K_{il} \f]
 * where \f$ {^{(3)}\Gamma^{k}_{ij}} \f$ is the spatial Christoffel symbol of
 * the second kind.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ijj<DataType, SpatialDim, Frame>
covariant_derivative_of_extrinsic_curvature(
    const tnsr::ijj<DataType, SpatialDim, Frame>& d_ex_curv,
    const tnsr::ii<DataType, SpatialDim, Frame>& ex_curv,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind);
template <typename DataType, size_t SpatialDim, typename Frame>
void covariant_derivative_of_extrinsic_curvature(
    gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*> grad_ex_curv,
    const tnsr::ijj<DataType, SpatialDim, Frame>& d_ex_curv,
    const tnsr::ii<DataType, SpatialDim, Frame>& ex_curv,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind);
/// @}

namespace Tags {
/// \copydoc covariant_derivative_of_extrinsic_curvature
template <size_t SpatialDim, typename Frame>
struct CovariantDerivativeOfExtrinsicCurvatureCompute
    : gr::Tags::CovariantDerivativeOfExtrinsicCurvature<DataVector, SpatialDim,
                                                        Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      ::Tags::deriv<gr::Tags::ExtrinsicCurvature<DataVector, SpatialDim, Frame>,
                    tmpl::size_t<SpatialDim>, Frame>,
      gr::Tags::ExtrinsicCurvature<DataVector, SpatialDim, Frame>,
      gr::Tags::SpatialChristoffelSecondKind<DataVector, SpatialDim, Frame>>;

  using return_type = tnsr::ijj<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::ijj<DataVector, SpatialDim, Frame>*>,
      const tnsr::ijj<DataVector, SpatialDim, Frame>&,
      const tnsr::ii<DataVector, SpatialDim, Frame>&,
      const tnsr::Ijj<DataVector, SpatialDim, Frame>&
          spatial_christoffel_second_kind)>(
      &covariant_derivative_of_extrinsic_curvature<DataVector, SpatialDim,
                                                   Frame>);

  using base =
      gr::Tags::CovariantDerivativeOfExtrinsicCurvature<DataVector, SpatialDim,
                                                        Frame>;
};
}  // namespace Tags
}  // namespace gr
