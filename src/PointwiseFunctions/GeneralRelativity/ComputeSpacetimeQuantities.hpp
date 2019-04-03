// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating spacetime tensors from 3+1 quantities

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the spacetime metric from the spatial metric, lapse, and
 * shift.
 * \details The spacetime metric \f$ \psi_{ab} \f$ is calculated as
 * \f{align}{
 *   \psi_{tt} &= - N^2 + N^m N^n g_{mn} \\
 *   \psi_{ti} &= g_{mi} N^m  \\
 *   \psi_{ij} &= g_{ij}
 * \f}
 * where \f$ N, N^i\f$ and \f$ g_{ij}\f$ are the lapse, shift and spatial metric
 * respectively
 */
template <size_t Dim, typename Frame, typename DataType>
void spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, Dim, Frame>*> spacetime_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept;
// @}

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial metric from spacetime metric.
 * \details Simply pull out the spatial components.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> spatial_metric(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute inverse spacetime metric from inverse spatial metric, lapse
 * and shift
 *
 * \details The inverse spacetime metric \f$ \psi^{ab} \f$ is calculated as
 * \f{align}
 *    \psi^{tt} &= -  1/N^2 \\
 *    \psi^{ti} &= N^i / N^2 \\
 *    \psi^{ij} &= g^{ij} - N^i N^j / N^2
 * \f}
 * where \f$ N, N^i\f$ and \f$ g^{ij}\f$ are the lapse, shift and inverse
 * spatial metric respectively
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::AA<DataType, SpatialDim, Frame> inverse_spacetime_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute shift from spacetime metric and inverse spatial metric.
 *
 * \details Computes
 * \f{align}
 *    N^i &= g^{ij} \psi_{jt}
 * \f}
 * where \f$ N^i\f$, \f$ g^{ij}\f$, and \f$\psi_{ab}\f$ are the shift, inverse
 * spatial metric, and spacetime metric.
 * This can be derived, e.g., from Eqs. 2.121--2.122 of Baumgarte & Shapiro.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> shift(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute lapse from shift and spacetime metric
 *
 * \details Computes
 * \f{align}
 *    N &= \sqrt{N^i \psi_{it}-\psi_{tt}}
 * \f}
 * where \f$ N \f$, \f$ N^i\f$, and \f$\psi_{ab}\f$ are the lapse, shift,
 * and spacetime metric.
 * This can be derived, e.g., from Eqs. 2.121--2.122 of Baumgarte & Shapiro.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> lapse(
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spacetime derivative of spacetime metric from spatial metric,
 * lapse, shift, and their space and time derivatives.
 *
 * \details Computes the derivatives as:
 * \f{align}
 *     \partial_\mu \psi_{tt} &= - 2 N \partial_\mu N
 *                 + 2 g_{mn} N^m \partial_\mu N^n
 *                 + N^m N^n \partial_\mu g_{mn} \\
 *     \partial_\mu \psi_{ti} &= g_{mi} \partial_\mu N^m
 *                 + N^m \partial_\mu g_{mi} \\
 *     \partial_\mu \psi_{ij} &= \partial_\mu g_{ij}
 * \f}
 * where \f$ N, N^i, g \f$ are the lapse, shift, and spatial metric
 * respectively.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::abb<DataType, SpatialDim, Frame> derivatives_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;

/*!
 * \brief Computes spacetime normal one-form from lapse.
 *
 * \details If \f$N\f$ is the lapse, then
 * \f{align} n_t &= - N \\
 * n_i &= 0 \f}
 * is computed.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_normal_one_form(
    const Scalar<DataType>& lapse) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes spacetime normal vector from lapse and shift.
 * \details If \f$N, N^i\f$ are the lapse and shift respectively, then
 * \f{align} n^t &= 1/N \\
 * n^i &= -\frac{N^i}{N} \f}
 * is computed.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::A<DataType, SpatialDim, Frame> spacetime_normal_vector(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes extrinsic curvature from metric and derivatives.
 * \details Uses the ADM evolution equation for the spatial metric,
 * \f[ K_{ij} = \frac{1}{2N} \left ( -\partial_0 g_{ij}
 * + N^k \partial_k g_{ij} + g_{ki} \partial_j N^k
 * + g_{kj} \partial_i N^k \right ) \f]
 * where \f$K_{ij}\f$ is the extrinsic curvature, \f$N\f$ is the lapse,
 * \f$N^i\f$ is the shift, and \f$g_{ij}\f$ is the spatial metric. In terms
 * of the Lie derivative of the spatial metric with respect to a unit timelike
 * vector \f$t^a\f$ normal to the spatial slice, this corresponds to the sign
 * convention
 * \f[ K_{ab} = - \frac{1}{2} \mathcal{L}_{\mathbf{t}} g_{ab} \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;

}  // namespace gr
