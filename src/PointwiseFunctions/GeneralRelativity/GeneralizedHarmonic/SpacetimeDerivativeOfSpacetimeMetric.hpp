// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the spacetime derivative of the spacetime metric,
 * \f$\partial_a g_{bc}\f$
 *
 * \f{align*}{
 * \partial_t g_{ab}&=-\alpha \Pi_{ab} + \beta^i \Phi_{iab} \\
 * \partial_i g_{ab}&=\Phi_{iab}
 * \f}
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void spacetime_derivative_of_spacetime_metric(
    gsl::not_null<tnsr::abb<DataType, SpatialDim, Frame>*> da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
}  // namespace gh
