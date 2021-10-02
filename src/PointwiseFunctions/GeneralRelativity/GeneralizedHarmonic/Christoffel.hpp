// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial Christoffel symbol of the 2nd kind from the
 * the generalized harmonic spatial derivative variable and the
 * inverse spatial metric.
 *
 * \details
 * If \f$ \Phi_{kab} \f$ is the generalized harmonic spatial derivative
 * variable \f$ \Phi_{kab} = \partial_k \psi_{ab}\f$ and \f$\gamma^{ij}\f$
 * is the inverse spatial metric, the Christoffel symbols are
 * \f[
 *   \Gamma^m_{ij} = \frac{1}{2}\gamma^{mk}(\Phi_{ijk}+\Phi_{jik}-\Phi_{kij}).
 * \f]
 *
 * In the not_null version, no memory allocations are performed if the
 * output tensor already has the correct size.
 *
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void christoffel_second_kind(
    const gsl::not_null<tnsr::Ijj<DataType, SpatialDim, Frame>*> christoffel,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inv_metric);

template <size_t SpatialDim, typename Frame, typename DataType>
auto christoffel_second_kind(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inv_metric)
    -> tnsr::Ijj<DataType, SpatialDim, Frame>;
/// @}
}  // namespace GeneralizedHarmonic
