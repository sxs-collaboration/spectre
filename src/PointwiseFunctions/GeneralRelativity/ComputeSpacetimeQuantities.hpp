// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating spacetime tensors from 3+1 quantities

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/*!
* \ingroup GeneralRelativityGroup
* \brief Computes the spacetime metric from the spatial metric, lapse, and
* shift.
* \details The spacetime metric \f$ \psi_{ab} \f$ is calculated as
* \f{align}{
*   \psi_{tt} &=& - N^2 + N^m N^n g_{mn} \\
*   \psi_{ti} &=& g_{mi} N^m  \\
*   \psi_{ij} &=& g_{ij}
* \f}
* where \f$ N, N^i\f$ and \f$ g_{ij}\f$ are the lapse, shift and spatial metric
* respectively
*/
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> compute_spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute inverse spacetime metric from inverse spatial metric, lapse
 * and shift
 *
 * \details The inverse spacetime metric \f$ \psi^{ab} \f$ is calculated as
 * \f{align}
 *    \psi^{tt} &=& -  1/N^2 \\
 *    \psi^{ti} &=& N^i / N^2 \\
 *    \psi^{ij} &=& g^{ij} - N^i N^j / N^2
 * \f}
 * where \f$ N, N^i\f$ and \f$ g^{ij}\f$ are the lapse, shift and inverse
 * spatial metric respectively
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::AA<DataType, SpatialDim, Frame> compute_inverse_spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;
