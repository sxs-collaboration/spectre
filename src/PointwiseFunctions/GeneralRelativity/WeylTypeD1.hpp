// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes a quantity measuring how far from type D spacetime is.
 *
 * \details Computes a quantity measuring how far from type D spacetime is,
 * using measure D1. Implements equation 8 of \cite Bhagwat2017tkm.
 *
 * \f{align}{
 * \frac{a}{12} \gamma_{ij} - \frac{b}{a} E_{ij} - 4
 E_{i}^{k} E_{jk} = 0 \f}
 *
 * where \f$\gamma_{ij}\f$ is the spatial metric, \f$E_{ij}\f$ is the
 * electric part ofthe Weyl tensor.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_type_D1_tensor(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_type_D1_tensor(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        weyl_type_D1_tensor,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the scalar of \f$D_{ij} D^{ij}\f$.
 *
 * \details Computes the scalar \f$D_{ij} D^{ij}\f$ (from equation 8 of
 * \cite Bhagwat2017tkm) from \f$D_{ij}\f$ and the inverse spatial metric
 * \f$\gamma^{ij}\f$, i.e. \f$D_{ij} = \gamma^{ik}\gamma^{jl}E_{ij}D_{kl}\f$.
 *
 * \note The electric part of the Weyl tensor in vacuum is available via
 * gr::weyl_electric(). The electric part of the Weyl tensor needs additional
 * terms for matter.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_type_D1_scalar(
    const gsl::not_null<Scalar<DataType>*> weyl_type_D1_tensor_scalar_result,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_type_D1_tensor,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weyl_type_D1_scalar(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_type_D1_tensor,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric);

/// @}

}  // namespace gr
