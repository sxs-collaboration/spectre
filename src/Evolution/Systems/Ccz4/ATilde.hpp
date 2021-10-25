// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \brief Computes the trace-free part of the extrinsic curvature
 *
 * \details Computes the trace-free part as:
 *
 * \f{align}
 *     \tilde A_{ij} &= \phi^2 \left(K_{ij} - \frac{1}{3} K \gamma_{ij}\right)
 * \f}
 *
 * where \f$\phi^2\f$ is the square of the conformal factor defined by
 * `Ccz4::Tags::ConformalFactorSquared`, \f$\gamma_{ij}\f$ is the spatial metric
 * defined by `gr::Tags::SpatialMetric`, \f$K_{ij}\f$ is the extrinsic curvature
 * defined by `gr::Tags::ExtrinsicCurvature`, and \f$K\f$ is the trace of the
 * extrinsic curvature defined by `gr::Tags::TraceExtrinsicCurvature`.
 */
template <size_t Dim, typename Frame, typename DataType>
void a_tilde(const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> result,
             const gsl::not_null<Scalar<DataType>*> buffer,
             const Scalar<DataType>& conformal_factor_squared,
             const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
             const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature,
             const Scalar<DataType>& trace_extrinsic_curvature);

template <size_t Dim, typename Frame, typename DataType>
tnsr::ii<DataType, Dim, Frame> a_tilde(
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature,
    const Scalar<DataType>& trace_extrinsic_curvature);
/// @}
}  // namespace Ccz4
