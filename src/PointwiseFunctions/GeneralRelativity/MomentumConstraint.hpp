// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
/// @{
/*!
 * \brief Computes the momentum constraint in vacuum.
 *
 * \details The momentum constraint in vacuum GR reads (cf. Eq. (2.96) in
 * \cite BaumgarteShapiro)
 * \begin{equation}
 *   \mathcal{M}_i = D_j {K^j}_i - D_i K = 0,
 * \end{equation}
 * where $D_i$ is the covariant derivative with respect to the spatial metric,
 * $K_{ij}$ is the extrinsic curvature and $K$ is its trace.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void momentum_constraint_in_vacuum(
    gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> momentum_constraint,
    const tnsr::ijj<DataType, SpatialDim, Frame>& d_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>& d_trace_extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::i<DataType, SpatialDim, Frame> momentum_constraint_in_vacuum(
    const tnsr::ijj<DataType, SpatialDim, Frame>& d_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>& d_trace_extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);
/// @}

}  // namespace gr
