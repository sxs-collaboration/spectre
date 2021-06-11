// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the time derivative of the spacetime metric from spatial
 * metric, lapse, shift, and their time derivatives.
 *
 * \details Computes the derivative as:
 *
 * \f{align}{
 * \partial_t g_{tt} &= - 2 \alpha \partial_t \alpha
 * - 2 \gamma_{i j} \beta^i \partial_t \beta^j
 * + \beta^i \beta^j \partial_t \gamma_{i j}\\
 * \partial_t g_{t i} &= \gamma_{j i} \partial_t \beta^j
 * + \beta^j \partial_t \gamma_{j i}\\
 * \partial_t g_{i j} &= \partial_t \gamma_{i j},
 * \f}
 *
 * where \f$\alpha, \beta^i, \gamma_{ij}\f$ are the lapse, shift, and spatial
 * metric respectively.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_derivative_of_spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> dt_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> time_derivative_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric) noexcept;
/// @}
namespace Tags {}  // namespace Tags
}  // namespace gr
