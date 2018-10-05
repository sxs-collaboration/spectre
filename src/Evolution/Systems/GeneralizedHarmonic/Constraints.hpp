// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate the generalized harmonic constraints

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic {
// @{
/*!
 * \brief Computes the generalized-harmonic 3-index constraint.
 *
 * \details Computes the generalized-harmonic 3-index constraint,
 * \f$C_{iab} = \partial_i\psi_{ab} - \Phi_{iab},\f$ which is
 * given by Eq. (26) of http://arXiv.org/abs/gr-qc/0512093v3
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> three_index_constraint(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void three_index_constraint(
    gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}
}  // namespace GeneralizedHarmonic
