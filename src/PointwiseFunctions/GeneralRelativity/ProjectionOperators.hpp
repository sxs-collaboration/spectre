// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute projection operator onto an interface
 *
 * \details Returns the operator \f$P^{ij} = g^{ij} - n^i n^j\f$,
 * where \f$g^{ij}\f$ is the inverse spatial metric, and
 * \f$n^i\f$ is the normal vector to the interface in question.
 *
 */
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::II<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void transverse_projection_operator(
    gsl::not_null<tnsr::II<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute projection operator onto an interface
 *
 * \details Returns the operator \f$P_{ij} = g_{ij} - n_i n_j\f$,
 * where \f$ g_{ij}\f$ is the spatial metric, and \f$ n_i\f$ is
 * the normal one-form to the interface in question.
 */
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::ii<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void transverse_projection_operator(
    gsl::not_null<tnsr::ii<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form) noexcept;
// @}

// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute projection operator onto an interface
 *
 * \details Returns the operator \f$P^{i}_{j} = \delta^{i}_{j} - n^i n_j\f$,
 * where \f$n^i\f$ and \f$n_i\f$ are the normal vector and normal one-form
 * to the interface in question.
 */
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::Ij<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void transverse_projection_operator(
    gsl::not_null<tnsr::Ij<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form) noexcept;
// @}
}  // namespace gr
