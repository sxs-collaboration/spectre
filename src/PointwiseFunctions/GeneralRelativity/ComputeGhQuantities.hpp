// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating spacetime tensors from 3+1 quantities

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace GeneralizedHarmonic {
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the auxiliary variable \f$\phi_{iab}\f$ used by the
 * generalized harmonic formulation of Einstein's equations.
 *
 * \details If \f$ N, N^i\f$ and \f$ g_{ij} \f$ are the lapse, shift and spatial
 * metric respectively, then \f$\phi_{iab} \f$ is computed as
 *
 * \f{align}
 *     \phi_{ktt} &= - 2 N \partial_k N
 *                 + 2 g_{mn} N^m \partial_k N^n
 *                 + N^m N^n \partial_k g_{mn} \\
 *     \phi_{kti} &= g_{mi} \partial_k N^m
 *                 + N^m \partial_k g_{mi} \\
 *     \phi_{kij} &= \partial_k g_{ij}
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
 * respectively, and \f$ \phi_{iab} = \partial_i \psi_{ab} \f$ then
 * \f$\Pi_{\mu\nu} = -(1/N) ( \partial_t \psi_{\mu\nu}  -
 *      N^m \phi_{m\mu\nu}) \f$ where \f$ \partial_t \psi_{ab} \f$ is computed
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

}  // namespace GeneralizedHarmonic
