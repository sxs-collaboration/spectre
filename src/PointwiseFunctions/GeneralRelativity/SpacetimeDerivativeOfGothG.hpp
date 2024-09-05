// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spacetime derivative of
 * \f$ \mathfrak{g}^{ab}\equiv (-g)^{1/2} g^{ab} \f$.
 *
 * \details Computes the spacetime derivative of
 * \f$ \mathfrak{g}^{ab}\equiv (-g)^{1/2} g^{ab} \f$, defined
 * in \cite Misner1973. Using \f$ (-g)^{1/2} = \alpha (\gamma)^{1/2} \f$, where
 * \f$ \alpha \f$ is the lapse and \f$ \gamma \f$ is the determinant of the
 * spatial metric (\cite BaumgarteShapiro), the derivative of
 * \f$ \mathfrak{g}^{ab} \f$ expands out to
 * \f{align}{
 *   \partial_c \mathfrak{g}^{ab} &= \left[\partial_c \alpha \gamma^{1/2}
 *   + \frac{1}{2} \alpha \partial_c \gamma \gamma^{-1/2} \right] g^{ab}
 *   - \alpha \gamma^{1/2} g^{ad} g^{be} \partial_c g_{de}.
 * \f}
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void spacetime_deriv_of_goth_g(
    gsl::not_null<tnsr::aBB<DataType, SpatialDim, Frame>*> da_goth_g,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::a<DataType, SpatialDim, Frame>& da_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::aBB<DataType, SpatialDim, Frame> spacetime_deriv_of_goth_g(
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::a<DataType, SpatialDim, Frame>& da_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric);
/// @}
}  // namespace gr
