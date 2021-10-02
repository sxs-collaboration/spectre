// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the propagating modes of the Weyl tensor
 *
 * \details The Weyl tensor evolution system in vacuum has six characteristic
 * fields, of which two (\f$ U^{8\pm}\f$) are proportional to the
 * Newman-Penrose components of the Weyl tensor \f$\Psi_4\f$ and \f$\Psi_0\f$.
 * These represent the true gravitational-wave degrees of freedom, and
 * can be written down in terms of \f$3+1\f$ quantities as
 * \cite Kidder2004rw (see Eq. 75):
 *
 * \f{align}
 * U^{8\pm}_{ij} &= \left(P^{k}_i P^{l}_j - \frac{1}{2} P_{ij} P^{kl}\right)
 *                  \left(R_{kl} + K K_{kl} - K_k^m K_{ml}
 *                       \mp n^m \nabla_m K_{kl} \pm n^m \nabla_{(k}K_{l)m}
 *                       \right),\\
 *               &= \left(P^{k}_i P^{l}_j - \frac{1}{2} P_{ij} P^{kl}\right)
 *                  \left(E_{kl} \mp n^m \nabla_m K_{kl}
 *                        \pm n^m \nabla_{(k}K_{l)m}\right),
 * \f}
 *
 * where \f$R_{ij}\f$ is the spatial Ricci tensor, \f$K_{ij}\f$ is the
 * extrinsic curvature, \f$K\f$ is the trace of \f$K_{ij}\f$, \f$E_{ij}\f$ is
 * the electric part of the Weyl tensor in vacuum, \f$n^i\f$ is the outward
 * directed unit normal vector to the interface, \f$\nabla_i\f$ denotes the
 * covariant derivative, and \f$P^{ij}\f$ and its index-raised and lowered forms
 * project tensors transverse to \f$n^i\f$.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_propagating(
    const tnsr::ii<DataType, SpatialDim, Frame>& ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::I<DataType, SpatialDim, Frame>& unit_interface_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& projection_IJ,
    const tnsr::ii<DataType, SpatialDim, Frame>& projection_ij,
    const tnsr::Ij<DataType, SpatialDim, Frame>& projection_Ij,
    const double sign);

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_propagating(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> weyl_prop_u8,
    const tnsr::ii<DataType, SpatialDim, Frame>& ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::I<DataType, SpatialDim, Frame>& unit_interface_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& projection_IJ,
    const tnsr::ii<DataType, SpatialDim, Frame>& projection_ij,
    const tnsr::Ij<DataType, SpatialDim, Frame>& projection_Ij,
    const double sign);
/// @}
}  // namespace gr
