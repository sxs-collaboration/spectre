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
 * \brief Computes time derivative of index lowered shift from generalized
 *        harmonic variables, spatial metric and its time derivative.
 *
 * \details The time derivative of \f$ \beta_i \f$ is given by:
 * \f{align*}
 *  \partial_0 \beta_i =
 *      \gamma_{ij} \partial_0 \beta^j + \beta^j \partial_0 \gamma_{ij}
 * \f}
 * where the first term is obtained from `time_deriv_of_shift()`, and the latter
 * is a user input.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void time_deriv_of_lower_shift(
    gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> dt_lower_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::i<DataType, SpatialDim, Frame> time_deriv_of_lower_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi);
/// @}
}  // namespace gh
