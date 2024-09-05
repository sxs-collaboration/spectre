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
 * \brief Computes the time derivative of the spacetime metric from the
 * generalized harmonic quantities \f$\Pi_{a b}\f$, \f$\Phi_{i a b}\f$, and the
 * lapse \f$\alpha\f$ and shift \f$\beta^i\f$.
 *
 * \details Computes the derivative as:
 *
 * \f{align}{
 * \partial_t g_{a b} = \beta^i \Phi_{i a b} - \alpha \Pi_{a b}.
 * \f}
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void time_derivative_of_spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> dt_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::aa<DataType, SpatialDim, Frame> time_derivative_of_spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
/// @}
}  // namespace gh
