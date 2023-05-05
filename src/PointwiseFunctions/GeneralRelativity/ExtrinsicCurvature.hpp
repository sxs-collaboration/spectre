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
}  // namespace gr
