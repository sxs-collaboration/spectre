// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace GeneralizedHarmonic {
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
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_derivative_of_spacetime_metric(
    gsl::not_null<tnsr::abb<DataType, SpatialDim, Frame>*> da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi);
}  // namespace GeneralizedHarmonic
