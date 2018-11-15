// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating spacetime tensors from 3+1 quantities

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace GeneralizedHarmonic {
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the auxiliary variable \f$\Phi_{iab}\f$ used by the
 * generalized harmonic formulation of Einstein's equations.
 *
 * \details If \f$ N, N^i\f$ and \f$ g_{ij} \f$ are the lapse, shift and spatial
 * metric respectively, then \f$\Phi_{iab} \f$ is computed as
 *
 * \f{align}
 *     \Phi_{ktt} &= - 2 N \partial_k N
 *                 + 2 g_{mn} N^m \partial_k N^n
 *                 + N^m N^n \partial_k g_{mn} \\
 *     \Phi_{kti} &= g_{mi} \partial_k N^m
 *                 + N^m \partial_k g_{mi} \\
 *     \Phi_{kij} &= \partial_k g_{ij}
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> phi(
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the conjugate momentum \f$\Pi_{ab}\f$ of the spacetime metric
 * \f$ \psi_{ab} \f$.
 *
 * \details If \f$ N, N^i\f$ are the lapse and shift
 * respectively, and \f$ \Phi_{iab} = \partial_i \psi_{ab} \f$ then
 * \f$\Pi_{\mu\nu} = -(1/N) ( \partial_t \psi_{\mu\nu}  -
 *      N^m \Phi_{m\mu\nu}) \f$ where \f$ \partial_t \psi_{ab} \f$ is computed
 * as
 *
 * \f{align}
 *     \partial_t \psi_{tt} &= - 2 N \partial_t N
 *                 + 2 g_{mn} N^m \partial_t N^n
 *                 + N^m N^n \partial_t g_{mn} \\
 *     \partial_t \psi_{ti} &= g_{mi} \partial_t N^m
 *                 + N^m \partial_t g_{mi} \\
 *     \partial_t \psi_{ij} &= \partial_t g_{ij}
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> pi(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief  Computes generalized harmonic gauge source function.
 * \details If \f$N, N^i, g_{ij}, \Gamma_{ijk}, K\f$ are the lapse, shift,
 * spatial metric, spatial Christoffel symbols, and trace of the extrinsic
 * curvature, then we compute
 * \f{align}
 * H_l &= N^{-2} g_{il}(\partial_t N^i - N^k \partial_k N^i)
 * + N^{-1} \partial_l N - g^{km}\Gamma_{lkm} \\
 * H_0 &= -N^{-1} \partial_t N + N^{-1} N^k\partial_k N + N^k H_k - N K
 * \f}
 * See Eqs. 8 and 9 of [arXiv:gr-qc/0512093]
 * (https://arxiv.org/abs/gr-qc/0512093)
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_source(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>&
        trace_christoffel_last_indices) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes extrinsic curvature from generalized harmonic variables
 *        and the spacetime normal vector.
 *
 * \details If \f$ \Pi_{ab} \f$ and \f$ \Phi_{iab} \f$ are the generalized
 * harmonic conjugate momentum and spatial derivative variables, and if
 * \f$t^a\f$ is the spacetime normal vector, then the extrinsic curvature
 * is computed as
 * \f{align}
 *     K_{ij} &= \frac{1}{2} \Pi_{ij} + \Phi_{(ij)a} t^a
 * \f}
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of the spatial metric from
 *        the generalized harmonic spatial derivative variable.
 *
 * \details If \f$ \Phi_{kab} \f$ is the generalized
 * harmonic spatial derivative variable, then the derivatives of the
 * spatial metric are
 * \f{align}
 *      \partial_k g_{ij} &= \Phi_{kij}
 * \f}
 *
 * This quantity is needed for computing spatial Christoffel symbols.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ijj<DataType, SpatialDim, Frame> deriv_spatial_metric(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes spatial derivatives of lapse (N) from the generalized
 *        harmonic variables and spacetime unit normal 1-form.
 *
 * \details If the generalized harmonic conjugate momentum and spatial
 * derivative variables are \f$\Pi_{ab} = -t^c \partial_c \psi_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i \psi_{ab} \f$, the spatial derivatives of N
 * can be obtained from:
 * \f{align*}
 *  t^a t^b \Phi_{iab} = -\frac{1}{2N} [\partial_i (-N^2 + N_jN^j)-
 *                               2 N^j \partial_i N_j
 *                               + N^j N^k \partial_i g_{jk}]
 *                     = -\frac{2}{N} \partial_i N,
 * \f}
 * since
 * \f[
 * \partial_i (N_jN^j) = 2N^j \partial_i N_j - N^j N^k \partial_i g_{jk}.
 * \f]
 *
 * \f[
 * \Longrightarrow \partial_i N = -(N/2) t^a \Phi_{iab} t^b
 * \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> spatial_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept;

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of lapse (N) from the generalized
 *        harmonic variables, lapse, shift and the spacetime unit normal 1-form.
 *
 * \details Let the generalized harmonic conjugate momentum and spatial
 * derivative variables be \f$\Pi_{ab} = -t^c \partial_c \psi_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i \psi_{ab} \f$, and the operator
 * \f$D := \partial_0 - N^k \partial_k \f$. The time derivative of N is then:
 * \f{align*}
 *  \frac{1}{2} N^2 t^a t^b \Pi_{ab} - \frac{1}{2} N N^i t^a t^b \Phi_{iab}
 *  =& \frac{1}{2} N^2 t^a t^b t^c \partial_c \psi_{ab}
 *       - \frac{1}{2} N N^i (-(2/N) \partial_i N) \\
 *  =& \frac{1}{2} N^2 [-(1/N^3) D[g_{jk} N^j N^k - N^2] \\
 *           &- (N^j N^k / N^3)D[g_{jk}] \\
 *           &+ 2 (N^j / N^3) D[g_{jk} N^k] + (2 / N^2)(N^i \partial_i N)] \\
 *  =& \frac{1}{2N} [-D[g_{jk}N^jN^k - N^2] - N^jN^k D[g_{jk}]
 *            + 2N N^k\partial_k N + 2N^j D[g_{jk}N^k]] \\
 *  =& D[N] + N^k\partial_k N \\
 *  =& \partial_0 N
 * \f}
 * where the simplification done for \f$\partial_i N\f$ is used to substitute
 * for the second term (\f$\frac{1}{2} N N^i t^a t^b \Phi_{iab}\f$).
 *
 * Thus,
 * \f[
 *  \partial_0 N = (N/2)(N t^a t^b \Pi_{ab} - N^i t^a t^b \Phi_{iab})
 * \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> time_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
}  // namespace GeneralizedHarmonic
