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
//@{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes extrinsic curvature from metric and derivatives.
 * \details Uses the ADM evolution equation for the spatial metric,
 * \f[ K_{ij} = \frac{1}{2N} \left ( -\partial_0 g_{ij}
 * + N^k \partial_k g_{ij} + g_{ki} \partial_j N^k
 * + g_{kj} \partial_i N^k \right ) \f]
 * where \f$K_{ij}\f$ is the extrinsic curvature, \f$N\f$ is the lapse,
 * \f$N^i\f$ is the shift, and \f$g_{ij}\f$ is the spatial metric. In terms
 * of the Lie derivative of the spatial metric with respect to a unit timelike
 * vector \f$t^a\f$ normal to the spatial slice, this corresponds to the sign
 * convention
 * \f[ K_{ab} = - \frac{1}{2} \mathcal{L}_{\mathbf{t}} g_{ab} \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void extrinsic_curvature(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> ex_curvature,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;
//@}
}  // namespace gr
