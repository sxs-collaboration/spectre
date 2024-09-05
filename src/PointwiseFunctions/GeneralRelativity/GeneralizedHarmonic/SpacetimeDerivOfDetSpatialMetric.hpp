// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spacetime derivatives of the determinant of spatial metric,
 *        using the generalized harmonic variables, spatial metric, and its
 *        time derivative.
 *
 * \details Using the relation
 * \f$ \partial_a \gamma = \gamma \gamma^{jk} \partial_a \gamma_{jk} \f$
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void spacetime_deriv_of_det_spatial_metric(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_det_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_det_spatial_metric(
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
/// @}
}  // namespace gh
