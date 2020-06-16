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
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of index lowered shift from generalized
 *        harmonic variables, spatial metric and its time derivative.
 *
 * \details The time derivative of \f$ N_i \f$ is given by:
 * \f{align*}
 *  \partial_0 N_i = g_{ij} \partial_0 N^j + N^j \partial_0 g_{ij}
 * \f}
 * where the first term is obtained from `time_deriv_of_shift()`, and the latter
 * is a user input.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_lower_shift(
    gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> dt_lower_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> time_deriv_of_lower_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
// @}
}  // namespace GeneralizedHarmonic
