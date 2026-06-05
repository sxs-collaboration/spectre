// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
/// @{
/*!
 * \brief Computes the hamiltonian constraint in vacuum.
 *
 * \details The hamiltonian constraint in vacuum GR reads (cf. Eq. (2.90) in
 * \cite BaumgarteShapiro)
 * \begin{equation}
 *   \mathcal{H} = R + K^2 - K_{ij}K^{ij} = 0,
 * \end{equation}
 * where $R$ is the spatial Ricci scalar, $K_{ij}$ is the extrinsic curvature
 * and $K$ is its trace.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void hamiltonian_constraint_in_vacuum(
    gsl::not_null<Scalar<DataType>*> hamiltonian_constraint,
    const Scalar<DataType>& ricci_scalar,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature);

template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> hamiltonian_constraint_in_vacuum(
    const Scalar<DataType>& ricci_scalar,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature);
/// @}
}  // namespace gr
