// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Declares function templates to calculate the Ricci tensor

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic {

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial Ricci tensor using evolved variables and
 * their first derivatives.
 *
 * \details Compute \f$R_{ij}\f$ as:
 *
 * \f{align}
 * R_{i j} =& \case{1}{2} g^{a b} \left( \partial_{(i} d_{a b j)}
 *            + \partial_a d_{(i j) b} - \partial_a d_{b i j}
 *            - \partial_{(i} d_{j) a b} \right) \\
 *          &+ \case{1}{2} b^a d_{a i j} - \case{1}{4} d^a d_{a i j}
 *            - b^a d_{(i j) a} - \case{1}{2} d_{a j}^{~ ~ b} d_{b i}^{~ ~ a}\\
 *          &+ \case{1}{2} d^a d_{(i j) a} + \case{1}{4} d_i^{~ a b} d_{j a b}
 *            + \case{1}{2} d_{~ ~ i}^{a b} d_{a b j},
 * \f}
 *
 * using the variable \f$d_{kij} \equiv \partial_k g_{ij}\f$ defined and
 * used in equations (2.13) - (2.20) of \cite Kidder:2001tz to derive
 * the equation above.
 */
template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_ricci_tensor(
    gsl::not_null<tnsr::ii<DataType, VolumeDim, Frame>*> ricci,
    const tnsr::iaa<DataType, VolumeDim, Frame>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame>& deriv_phi,
    const tnsr::II<DataType, VolumeDim, Frame>&
        inverse_spatial_metric) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::ii<DataType, VolumeDim, Frame> spatial_ricci_tensor(
    const tnsr::iaa<DataType, VolumeDim, Frame>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame>& deriv_phi,
    const tnsr::II<DataType, VolumeDim, Frame>&
        inverse_spatial_metric) noexcept;
// @}
}  // namespace GeneralizedHarmonic
