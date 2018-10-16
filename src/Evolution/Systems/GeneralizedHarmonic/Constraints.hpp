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

// @{
/*!
 * \brief Computes the generalized-harmonic gauge constraint.
 *
 * \details Computes the generalized-harmonic gauge constraint
 * [Eq. (40) of http://arXiv.org/abs/gr-qc/0512093v3],
 * \f[
 * C_a = H_a + g^{ij} \Phi_{ija} + t^b \Pi_{ba}
 * - \frac{1}{2} g^i_a \psi^{bc} \Phi_{ibc}
 * - \frac{1}{2} t_a \psi^{bc} \Pi_{bc},
 * \f]
 * where \f$H_a\f$ is the gauge function,
 * \f$\psi_{ab}\f$ is the spacetime metric,
 * \f$\Pi_{ab}=-t^c\partial_c \psi_{ab}\f$, and
 * \f$\Phi_{iab} = \partial_i\psi_{ab}\f$; \f$t^a\f$ is the timelike unit
 * normal vector to the spatial slice, \f$g^{ij}\f$ is the inverse spatial
 * metric, and \f$g^b_c = \delta^b_c + t^b t_c\f$.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void gauge_constraint(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;
// @}
}  // namespace GeneralizedHarmonic
