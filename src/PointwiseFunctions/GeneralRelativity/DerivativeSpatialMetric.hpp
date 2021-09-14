// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the spatial derivative of the inverse spatial metric from the
 * inverse spatial metric and the spatial derivative of the spatial metric.
 *
 * \details Computes the derivative as:
 * \f{align}
 *     \partial_k \gamma^{ij} &= -\gamma^{in} \gamma^{mj}
 *                 \partial_k \gamma_{nm}
 * \f}
 * where \f$\gamma^{ij}\f$ and \f$\partial_k \gamma_{ij}\f$ are the inverse
 * spatial metric and spatial derivative of the spatial metric, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
void deriv_inverse_spatial_metric(
    const gsl::not_null<tnsr::iJJ<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric) noexcept;

template <size_t Dim, typename Frame, typename DataType>
tnsr::iJJ<DataType, Dim, Frame> deriv_inverse_spatial_metric(
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& d_spatial_metric) noexcept;
/// @}
}  // namespace gr
