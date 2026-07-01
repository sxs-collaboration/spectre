// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarTensor::sgb {
/// @{
/*!
 * \brief Computes the projection of the second covariant derivative of the
 * coupling function onto the normal vector.
 *
 * \details Computes the term
 * \begin{equation}
 *   n^a n^b \nabla_a\nabla_b F[\Psi]
 *     = \frac{\delta^2 F[\Psi]}{\delta \Psi^2} \Pi^2
 *     + \frac{\delta F[\Psi]}{\delta \Psi} n^a n^b \nabla_a \nabla_b \Psi,
 * \end{equation}
 * where $\Psi$ is the scalar field, $\Pi$ is its conjugate momentum, $F[\Psi]$
 * is the coupling function and $n^a$ is the unit vector normal to
 * the spatial hypersurfaces.
 */
template <typename DataType>
void DDCoupling_normal_normal_projection(
    gsl::not_null<Scalar<DataType>*> DDCoupling_normal_normal_result,
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const Scalar<DataType>& pi_scalar,
    const Scalar<DataType>& normal_normal_DD_scalar);

template <typename DataType>
Scalar<DataType> DDCoupling_normal_normal_projection(
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const Scalar<DataType>& pi_scalar,
    const Scalar<DataType>& normal_normal_DD_scalar);
/// @}

/// @{
/*!
 * \brief Computes the mixed projection of the second covariant derivative of
 * the coupling function.
 *
 * \details Computes the term
 * \begin{equation}
 *   \gamma^a_i n^b \nabla_a \nabla_b F[\Psi]
 *     = - \frac{\delta^2 F[\Psi]}{\delta \Psi^2} \Pi \partial_i \Psi
 *     + \frac{\delta F[\Psi]}{\delta \Psi}
 *       \gamma^a_i n^b \nabla_a \nabla_b \Psi,
 * \end{equation}
 * where $\Psi$ is the scalar field, $\Pi$ is its conjugate momentum and
 * $F[\Psi]$ is the coupling function; $n^a$ is the unit vector normal to the
 * spatial hypersurfaces and $\gamma^a_b = \delta^a_b + n^a n_b$ is the
 * projection operator onto them.
 */
template <typename DataType, typename Frame>
void DDCoupling_normal_spatial_projection(
    gsl::not_null<tnsr::i<DataType, 3, Frame>*>
        DDCoupling_normal_spatial_result,
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const Scalar<DataType>& pi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_scalar_field,
    const tnsr::i<DataType, 3, Frame>& normal_spatial_DD_scalar);

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame> DDCoupling_normal_spatial_projection(
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const Scalar<DataType>& pi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_scalar_field,
    const tnsr::i<DataType, 3, Frame>& normal_spatial_DD_scalar);
/// @}

/// @{
/*!
 * \brief Computes the spatial projection of the second covariant derivative of
 * the coupling function.
 *
 * \details Computes the term
 * \begin{equation}
 *   \gamma^a_i \gamma^b_j \nabla_a \nabla_b F[\Psi]
 *     = \frac{\delta^2 F[\Psi]}{\delta \Psi^2} (\partial_i \Psi)
 *       (\partial_j \Psi)
 *     + \frac{\delta F[\Psi]}{\delta \Psi}
 *       \gamma^a_i \gamma^b_j \nabla_a \nabla_b \Psi,
 * \end{equation}
 * where $\Psi$ is the scalar field, $\Pi$ is its conjugate momentum and
 * $F[\Psi]$ is the coupling function; $n^a$ is the unit vector normal to the
 * spatial hypersurfaces and $\gamma^a_b = \delta^a_b + n^a n_b$ is the
 * projection operator onto them.
 */
template <typename DataType, typename Frame>
void DDCoupling_spatial_spatial_projection(
    gsl::not_null<tnsr::ii<DataType, 3, Frame>*>
        DDCoupling_spatial_spatial_result,
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const tnsr::i<DataType, 3, Frame>& d_scalar_field,
    const tnsr::ii<DataType, 3, Frame>& spatial_spatial_DD_scalar);

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame> DDCoupling_spatial_spatial_projection(
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const tnsr::i<DataType, 3, Frame>& d_scalar_field,
    const tnsr::ii<DataType, 3, Frame>& spatial_spatial_DD_scalar);
/// @}
}  // namespace ScalarTensor::sgb
