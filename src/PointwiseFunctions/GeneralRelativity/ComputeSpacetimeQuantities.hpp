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
template <size_t Dim, typename Fr, typename DataType>
tnsr::aa<DataType, Dim, Fr> compute_spacetime_metric(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift,
    const tnsr::ii<DataType, Dim, Fr>& spatial_metric) noexcept;
