// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the time derivative of the spatial metric from extrinsic
 * curvature, lapse, shift, and their time derivatives.
 *
 * \details Computes the derivative as (see e.g. \cite BaumgarteShapiro, Eq.
 * (2.134)):
 *
 * \f{equation}
 * \partial_t \gamma_{ij} = -2 \alpha K_{ij} + 2 \nabla_{(i} \beta_{j)}
 *   = -2 \alpha K_{ij} + \beta^k \partial_k \gamma_{ij}
 *     + \gamma_{ik} \partial_j \beta^k + \gamma_{jk} \partial_i \beta^k
 * \f}
 *
 * where \f$\alpha\f$ is the lapse, \f$\beta^i\f$ is the shift,
 * \f$\gamma_{ij}\f$ is the spatial metric and \f$K_{ij}\f$ is the extrinsic
 * curvature.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_derivative_of_spatial_metric(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> dt_spatial_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature);

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> time_derivative_of_spatial_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature);
/// @}
}  // namespace gr
