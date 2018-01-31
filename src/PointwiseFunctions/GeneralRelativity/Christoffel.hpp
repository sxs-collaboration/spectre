// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate Christoffel symbols

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace gr {
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Christoffel symbol of the first kind from derivative of
 * metric
 *
 * \details Computes Christoffel symbol \f$\Gamma_{abc}\f$ as:
 * \f$ \Gamma_{cab} = 1/2 ( \partial_a g_{bc} + \partial_b g_{ac} -  \partial_c
 *    g_{ab}) \f$
 * where \f$g_{bc}\f$ is either a spatial or spacetime metric
 */
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::abb<DataType, SpatialDim, Frame, Index> christoffel_first_kind(
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric);
}  // namespace gr
