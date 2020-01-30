// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

// @{
/*!
 * \ingroup TensorGroup
 *
 * \brief Compute a spatial one-form orthonormal to the given unit form
 *
 * Given a unit spatial one-form \f$s_i\f$, compute a new form \f$t_i\f$
 * which is orthonormal to \f$s_i\f$, in the sense that
 * \f$\gamma^{ij}s_i t_j = 0\f$, for the given inverse spatial metric
 * \f$\gamma^{ij}\f$. The normalization of \f$t_i\f$ is such that
 * \f$\gamma^{ij}t_it_j = 1\f$.
 *
 * \details The new form is obtained via Gram-Schmidt process, starting
 * from a form whose components are \f$t_i = \delta_i^I\f$, where \f$I\f$ is
 * the index of the component of \f$s_i\f$ with the smallest absolute value.
 */
template <typename DataType, size_t VolumeDim, typename Frame>
void orthonormal_oneform(
    gsl::not_null<tnsr::i<DataType, VolumeDim, Frame>*> orthonormal_form,
    const tnsr::i<DataType, VolumeDim, Frame>& unit_form,
    const tnsr::II<DataType, VolumeDim, Frame>& inv_spatial_metric) noexcept;

template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::i<DataType, VolumeDim, Frame> orthonormal_oneform(
    const tnsr::i<DataType, VolumeDim, Frame>& unit_form,
    const tnsr::II<DataType, VolumeDim, Frame>& inv_spatial_metric) noexcept;
// @}

// @{
/*!
 * \ingroup TensorGroup
 *
 * \brief Compute a spatial one-form orthonormal to two given unit forms.
 *
 * Given a unit spatial one-form \f$s_i\f$ and another form \f$t_i\f$ that is
 * orthonormal to \f$s_i\f$, compute a new form \f$u_i\f$ which is orthonormal
 * to both \f$s_i\f$ and \f$t_i\f$, in the sense that
 * \f$\gamma^{ij}s_i u_j = \gamma^{ij}t_i u_j = 0\f$, for the given
 * inverse spatial metric \f$\gamma^{ij}\f$. The normalization of \f$u_i\f$
 * is such that \f$\gamma^{ij}u_iu_j = 1\f$.
 *
 * \details The new form is obtained by taking the covariant cross product
 * of \f$s_i\f$ and \f$ t_i\f$, for which the spatial metric as well as
 * its determinant must be provided.
 */
template <typename DataType, typename Frame>
void orthonormal_oneform(
    gsl::not_null<tnsr::i<DataType, 3, Frame>*> orthonormal_form,
    const tnsr::i<DataType, 3, Frame>& first_unit_form,
    const tnsr::i<DataType, 3, Frame>& second_unit_form,
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const Scalar<DataType>& det_spatial_metric) noexcept;

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame> orthonormal_oneform(
    const tnsr::i<DataType, 3, Frame>& first_unit_form,
    const tnsr::i<DataType, 3, Frame>& second_unit_form,
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const Scalar<DataType>& det_spatial_metric) noexcept;
// @}
