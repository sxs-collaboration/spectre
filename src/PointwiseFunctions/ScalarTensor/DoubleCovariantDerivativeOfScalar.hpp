// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarTensor {
/// @{
/*!
 * \brief Normal projection of the second covariant derivative of the scalar
 * field.
 *
 * \details Computes the term
 * \begin{equation}
 *   n^a n^b \nabla_a \nabla_b \Psi = - \frac{1}{\alpha} \Bigl[ \partial_t \Pi
 *      - \beta^i \partial_i \Pi
 *      + \Phi^i \partial_i \alpha \Bigr],
 * \end{equation}
 * where $\Psi$ is the scalar field, $\Pi$ is its conjugate momentum and
 * $\Phi_i = \partial_i \Psi$; $n^a$ is the unit vector normal to the spatial
 * hypersurfaces, while $\alpha$ is the lapse and $\beta^i$ is the shift vector.
 */
template <typename DataType, typename Frame>
void DDKG_normal_normal_projection(
    gsl::not_null<Scalar<DataType>*> DDKG_normal_normal_result,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, 3, Frame>& shift,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_pi_scalar,
    const Scalar<DataType>& dt_pi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_lapse);

template <typename DataType, typename Frame>
Scalar<DataType> DDKG_normal_normal_projection(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, 3, Frame>& shift,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_pi_scalar,
    const Scalar<DataType>& dt_pi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_lapse);
/// @}

/// @{
/*!
 * \brief Mixed projection of the second covariant derivative of the scalar
 * field.
 *
 * \details Computes the term
 * \begin{equation}
 *   \gamma^a_i n^b \nabla_a \nabla_b \Psi
 *     = - \partial_i \Pi + K_{ij} \Phi^j,
 * \end{equation}
 * where $\Psi$ is the scalar field, $\Pi$ is its conjugate momentum and
 * $\Phi_i = \partial_i \Psi$; $n^a$ is the unit vector normal to the spatial
 * hypersurfaces, $\gamma^a_b = \delta^a_b + n^a n_b$ is the projection operator
 * onto them and $K_{ij}$ is the extrinsic curvature.
 */
template <typename DataType, typename Frame>
void DDKG_normal_spatial_projection(
    gsl::not_null<tnsr::i<DataType, 3, Frame>*> DDKG_normal_spatial_result,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, 3, Frame>& extrinsic_curvature,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_pi_scalar);

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame> DDKG_normal_spatial_projection(
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, 3, Frame>& extrinsic_curvature,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_pi_scalar);
/// @}

/// @{
/*!
 * \brief Spatial projection of the second covariant derivative of the scalar
 * field.
 *
 * \details Computes the term
 * \begin{equation}
 *   \gamma^a_i \gamma^b_j \nabla_a \nabla_b \Psi =
 *     - \Pi K_{ij} + D_{(i} \Phi_{j)},
 * \end{equation}
 * where $\Psi$ is the scalar field, $\Pi$ is its conjugate momentum and
 * $\Phi_i = \partial_i \Psi$; $n^a$ is the unit vector normal to the spatial
 * hypersurfaces and $\gamma^a_b = \delta^a_b + n^a n_b$ is the projection
 * operator onto them; $K_{ij}$ is the extrinsic curvature and $D_i$ is the
 * covariant derivative with respect to the spatial metric.
 */
template <typename DataType, typename Frame>
void DDKG_spatial_spatial_projection(
    gsl::not_null<tnsr::ii<DataType, 3, Frame>*> DDKG_spatial_spatial_result,
    const tnsr::ii<DataType, 3, Frame>& extrinsic_curvature,
    const tnsr::Ijj<DataType, 3, Frame>& spatial_christoffel_second_kind,
    const Scalar<DataType>& pi_scalar,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::ij<DataType, 3, Frame>& d_phi_scalar);

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame> DDKG_spatial_spatial_projection(
    const tnsr::ii<DataType, 3, Frame>& extrinsic_curvature,
    const tnsr::Ijj<DataType, 3, Frame>& spatial_christoffel_second_kind,
    const Scalar<DataType>& pi_scalar,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::ij<DataType, 3, Frame>& d_phi_scalar);
/// @}
}  // namespace ScalarTensor
